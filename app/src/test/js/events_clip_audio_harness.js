'use strict';

// Run with: node app/src/test/js/events_clip_audio_harness.js
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const source = fs.readFileSync(path.join(__dirname, '../../main/assets/web/shared/events.js'), 'utf8');
const flush = () => new Promise(resolve => setImmediate(resolve));

function fixture(saved, volume) {
    const requests = [], plays = [], timers = new Map();
    const storage = new Map();
    if (saved !== null) storage.set('byd.events.unmuted', saved);
    if (volume !== undefined) storage.set('byd.events.volume', volume);
    let timerId = 0;
    const context = {
        localStorage: { getItem: key => storage.has(key) ? storage.get(key) : null },
        fetch: url => new Promise(resolve => requests.push({ url, resolve })),
        setTimeout(fn) { const id = ++timerId; timers.set(id, fn); return id; },
        clearTimeout(id) { timers.delete(id); }
    };
    context.window = context;
    vm.runInNewContext(source, context);
    const player = {
        muted: true, volume: 1, paused: true,
        play() {
            return new Promise((resolve, reject) => plays.push({ muted: this.muted, volume: this.volume, resolve, reject }));
        }
    };
    return { events: context.BYD.events, player, requests, plays, storage, timers };
}
const answer = (request, enabled) => request.resolve({ json: async () => ({ success: true, enabled }) });

async function run() {
    // Saved preferences are applied synchronously before the initial play request.
    for (const saved of ['0', '1']) {
        const f = fixture(saved, '45');
        f.events._startClipPlayback(f.player);
        assert.strictEqual(f.requests.length, 0);
        assert.strictEqual(f.plays.length, 1);
        assert.strictEqual(f.plays[0].muted, saved === '0');
        assert.strictEqual(f.plays[0].volume, 0.45);
        f.plays[0].resolve();
        await flush();
        assert.strictEqual(f.events._clipAudioStart, null);
    }
    // First review uses the recording setting before attempting playback.
    for (const enabled of [true, false]) {
        const f = fixture(null);
        f.events._startClipPlayback(f.player);
        assert.strictEqual(f.plays.length, 0);
        assert.strictEqual(f.requests[0].url, '/api/settings/audio-recording');
        answer(f.requests[0], enabled);
        await flush();
        assert.strictEqual(f.plays[0].muted, !enabled);
        assert.strictEqual(f.timers.size, 0);
    }
    // Only a policy rejection retries muted; the preference stays audible.
    const f = fixture('1');
    f.events._startClipPlayback(f.player);
    f.plays[0].reject({ name: 'NotAllowedError' });
    await flush();
    assert.deepStrictEqual(f.plays.map(p => p.muted), [false, true]);
    assert.strictEqual(f.storage.get('byd.events.unmuted'), '1');
    const g = fixture('1');
    g.events._startClipPlayback(g.player);
    g.plays[0].reject({ name: 'NotSupportedError' });
    await flush();
    assert.strictEqual(g.plays.length, 1);
    // A prior clip's asynchronous settings or rejection cannot affect the next clip.
    const h = fixture(null);
    h.events._startClipPlayback(h.player);
    h.storage.set('byd.events.unmuted', '0');
    h.events._startClipPlayback(h.player);
    answer(h.requests[0], true);
    await flush();
    assert.deepStrictEqual(h.plays.map(p => p.muted), [true]);
    const i = fixture('1');
    i.events._startClipPlayback(i.player);
    i.storage.set('byd.events.unmuted', '0');
    i.events._startClipPlayback(i.player);
    i.plays[0].reject({ name: 'NotAllowedError' });
    await flush();
    assert.deepStrictEqual(i.plays.map(p => p.muted), [false, true]);
    // Manual audio control wins over the pending first-run lookup.
    const j = fixture(null);
    j.events._startClipPlayback(j.player);
    j.player.muted = false;
    j.events._takeClipAudioControl(j.player);
    answer(j.requests[0], false);
    await flush();
    assert.deepStrictEqual(j.plays.map(p => p.muted), [false]);
    assert.strictEqual(j.player.muted, false);
    // Closing or manually taking play/pause control invalidates the pending startup.
    const k = fixture(null);
    k.events._startClipPlayback(k.player);
    k.events._clipAudioStart = null;
    answer(k.requests[0], true);
    await flush();
    assert.strictEqual(k.plays.length, 0);
    // A slow settings request cannot prevent playback indefinitely.
    const l = fixture(null);
    l.events._startClipPlayback(l.player);
    [...l.timers.values()][0]();
    await flush();
    assert.deepStrictEqual(l.plays.map(p => p.muted), [true]);
    answer(l.requests[0], true);
    await flush();
    assert.strictEqual(l.player.muted, true);
    assert.strictEqual(l.plays.length, 1);
    console.log('events_clip_audio_harness: ALL CHECKS PASSED');
}
run().catch(error => { console.error(error); process.exitCode = 1; });
