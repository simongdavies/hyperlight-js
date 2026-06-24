// Tests for the `eval` REPL primitive on LoadedJSSandbox.
import { describe, it, expect, beforeEach } from 'vitest';
import { SandboxBuilder } from '../lib.js';
import { expectRejectsWithCode } from './test-helpers.js';

async function loadedSandbox() {
    const proto = await new SandboxBuilder().build();
    const sandbox = await proto.loadRuntime();
    // No handler required — eval drives the persistent context directly.
    return sandbox.getLoadedSandbox();
}

describe('LoadedJSSandbox.eval', () => {
    let loaded;

    beforeEach(async () => {
        loaded = await loadedSandbox();
    });

    it('evaluates an expression', async () => {
        expect(await loaded.eval('1 + 2')).toBe(3);
    });

    it('marshals objects to JS values', async () => {
        expect(await loaded.eval('({ a: 1, b: [2, 3] })')).toEqual({
            a: 1,
            b: [2, 3],
        });
    });

    it('returns null for undefined completion values', async () => {
        expect(await loaded.eval('undefined')).toBeNull();
    });

    it('returns null for non-serializable (function) values', async () => {
        expect(await loaded.eval('(function () {})')).toBeNull();
    });

    it('persists top-level let/const across calls', async () => {
        await loaded.eval('let a = 10; const b = 5;');
        expect(await loaded.eval('a + b')).toBe(15);
    });

    it('persists function declarations across calls', async () => {
        await loaded.eval('function sq(x) { return x * x; }');
        expect(await loaded.eval('sq(9)')).toBe(81);
    });

    it('accumulates mutated state across calls', async () => {
        await loaded.eval('let n = 0;');
        await loaded.eval('n += 2;');
        await loaded.eval('n += 3;');
        expect(await loaded.eval('n')).toBe(5);
    });

    it('resolves an immediately-settling promise', async () => {
        expect(await loaded.eval('Promise.resolve(42)')).toBe(42);
    });

    it('shares state with registered handlers', async () => {
        // A fresh sandbox with a handler; eval-defined state is visible to it.
        const proto = await new SandboxBuilder().build();
        const sandbox = await proto.loadRuntime();
        sandbox.addHandler('read', 'function handler() { return globalThis.shared; }');
        const l = await sandbox.getLoadedSandbox();
        await l.eval('globalThis.shared = 123;');
        expect(await l.callHandler('read', {})).toBe(123);
    });

    it('throws on guest exceptions', async () => {
        await expect(loaded.eval("throw new Error('kaboom')")).rejects.toThrow(/kaboom/);
    });

    it('rejects empty code with ERR_INVALID_ARG', async () => {
        await expectRejectsWithCode(loaded.eval(''), 'ERR_INVALID_ARG');
    });

    it('rejects out-of-range wallClockTimeoutMs', async () => {
        await expectRejectsWithCode(loaded.eval('1', { wallClockTimeoutMs: 0 }), 'ERR_INVALID_ARG');
    });

    it('succeeds under a generous wall-clock monitor', async () => {
        expect(await loaded.eval('6 * 7', { wallClockTimeoutMs: 5000 })).toBe(42);
    });

    it('terminates an infinite loop via wall-clock monitor', async () => {
        await expectRejectsWithCode(
            loaded.eval('while (true) {}', { wallClockTimeoutMs: 200 }),
            'ERR_CANCELLED'
        );
        expect(loaded.poisoned).toBe(true);
    });
});
