"""CPU-only handler regression tests; run with pytest tests/test_demo_seed.py.

Compile unchanged function nodes from the working tree, avoiding app's Gradio
UI and inference's model/codec imports. Only ENGINE and WAV output are doubles;
_seed, set_seed and the three generation handlers execute their original bodies.
This does not exercise Gradio dispatch, model inference, CUDA or audio encoding.
"""
import ast
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('clone_voice', 'design_voice', 'edit_speech')
TIMEOUT = 10


def load_functions(filename, names, namespace):
    path = ROOT / filename
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


@pytest.fixture
def app():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    # Restore CUDA RNG too if this CPU test is run in an initialized GPU process.
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        inference = {'torch': torch, 'np': np, 'random': random}
        load_functions('inference.py', ['set_seed'], inference)
        namespace = {
            'random': random,
            'MAX_SEED': np.iinfo(np.int32).max,
            'fra': SimpleNamespace(set_seed=inference['set_seed']),
            'INFER_LOCK': threading.Lock(),
            'gr': SimpleNamespace(Error=ValueError),
            '_write_wav': lambda audio: audio,
        }
        load_functions('app.py', ['_seed', *NAMES], namespace)
        yield namespace
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def samples():
    return torch.rand(1).item(), float(np.random.rand()), random.random()


def baseline(app, seed):
    app['fra'].set_seed(seed)
    return samples()


def invoke(app, name, seed=123, randomize=False):
    inputs = {
        'clone_voice': dict(reference_audio='reference.wav', reference_text=' ref ',
                            target_text=' target ', language='en'),
        'design_voice': dict(instruction=' voice ', text=' target ', max_new_text_tokens=61),
        'edit_speech': dict(audio='input.wav', instruction=' edit ', edit_type='acoustic',
                            max_new_text_tokens=61),
    }
    return app[name](**inputs[name], seed=seed, randomize_seed=randomize,
                     inference_cfg=1.5, n_timesteps=7, max_new_audio_steps=42)


def engine(callback):
    return SimpleNamespace(**{
        method: callback for method in ('tts', 'voice_design', 'edit')
    })


class ObservedLock:
    """Signal a contended acquisition before blocking on a real threading.Lock."""
    def __init__(self):
        self.lock = threading.Lock()
        self.waiting = threading.Event()

    def __enter__(self):
        if not self.lock.acquire(blocking=False):
            self.waiting.set()
            if not self.lock.acquire(timeout=TIMEOUT):
                raise TimeoutError('inference lock acquisition timed out')
        return self

    def __exit__(self, *exc):
        self.lock.release()


@pytest.mark.parametrize('first,second', [
    ('clone_voice', 'design_voice'),
    ('design_voice', 'edit_speech'),
    ('edit_speech', 'clone_voice'),
])
def test_waiting_request_preserves_random_stream(app, first, second):
    expected = [baseline(app, seed) for seed in (123, 456)]
    lock = ObservedLock()
    app['INFER_LOCK'] = lock
    entered = threading.Event()
    release = threading.Event()

    def infer(**kwargs):
        if not entered.is_set():
            entered.set()
            if not release.wait(TIMEOUT):
                raise TimeoutError('first inference was not released')
        return SimpleNamespace(audio=samples(), text='generated')

    app['ENGINE'] = engine(infer)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(invoke, app, first, 123)
        try:
            assert entered.wait(TIMEOUT), 'first request never entered inference'
            b = pool.submit(invoke, app, second, 456)
            assert lock.waiting.wait(TIMEOUT), 'second request never tried the lock'
        finally:
            release.set()
        results = [a.result(timeout=TIMEOUT), b.result(timeout=TIMEOUT)]
    assert [r[-1] for r in results] == [123, 456]
    assert [r[0] for r in results] == expected


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('seed', [123, -1, 2**31 + 9])
def test_fixed_seed_and_model_arguments(app, name, seed):
    used = int(seed) % (app['MAX_SEED'] + 1)
    expected = baseline(app, used)
    calls = []

    def infer(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(audio=samples(), text='generated')

    app['ENGINE'] = engine(infer)
    result = invoke(app, name, seed)
    assert result == ((expected, used) if name == 'clone_voice'
                      else (expected, 'generated', used))
    expected_args = dict(n_timesteps=7, inference_cfg=1.5, max_new_audio_steps=42)
    expected_args.update({
        'clone_voice': dict(prompt_audio='reference.wav', prompt_text='ref',
                            target_text='target', language='en'),
        'design_voice': dict(instruction='voice', text='target', max_new_text_tokens=61),
        'edit_speech': dict(audio_path='input.wav', instruction='edit',
                            edit_type='acoustic', max_new_text_tokens=61),
    }[name])
    assert calls == [expected_args]


@pytest.mark.parametrize('name', NAMES)
def test_random_seed_is_selected_and_applied_under_lock(app, name, monkeypatch):
    lock = app['INFER_LOCK']
    expected = baseline(app, 789)
    original = app['fra'].set_seed
    seeded = []

    def randint(low, high):
        assert lock.locked()
        assert (low, high) == (0, app['MAX_SEED'])
        return 789

    def set_seed(seed):
        assert lock.locked()
        seeded.append(seed)
        original(seed)

    monkeypatch.setattr(random, 'randint', randint)
    monkeypatch.setattr(app['fra'], 'set_seed', set_seed)
    app['ENGINE'] = engine(lambda **kw: SimpleNamespace(audio=samples(), text=None))
    result = invoke(app, name, randomize=True)
    assert result[-1] == 789
    assert result[0] == expected
    assert seeded == [789]
    if name != 'clone_voice':
        assert result[1] == ''


@pytest.mark.parametrize('name', NAMES)
def test_inference_exception_releases_lock(app, name):
    def fail(**kwargs):
        raise RuntimeError('model failed')

    app['ENGINE'] = engine(fail)
    with pytest.raises(RuntimeError, match='model failed'):
        invoke(app, name)
    assert not app['INFER_LOCK'].locked()
    app['ENGINE'] = engine(lambda **kw: SimpleNamespace(audio=samples(), text='ok'))
    expected = baseline(app, 456)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(invoke, app, name, 456).result(timeout=TIMEOUT)
    assert result[0] == expected
    assert result[-1] == 456
