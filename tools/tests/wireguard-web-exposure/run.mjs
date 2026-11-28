import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

const source = readFileSync(new URL('../../../app/src/main/assets/web/shared/wireguard.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../../../app/src/main/assets/web/local/wireguard.html', import.meta.url), 'utf8');
const strings = JSON.parse(readFileSync(new URL('../../../app/src/main/assets/web/i18n/en.json', import.meta.url), 'utf8'));
assert.match(html, /id="wgExpose" disabled/);
assert.match(html, /onchange="WG.changeExposure\(\)"/);
assert.match(html, /wireguard\.js\?v=2/);
for (const key of ['expose_label', 'expose_warning', 'expose_apply', 'expose_on', 'expose_off', 'expose_saved', 'expose_failed']) {
    assert.ok(strings.wireguard[key], key);
}

const input = { checked: false, disabled: true };
const state = {};
const pending = [];
const toasts = [];
const sandbox = {
    document: { getElementById: id => ({ wgExpose: input, wgExposeState: state })[id] || null },
    window: {},
    fetch: (url, options) => new Promise((resolve, reject) => pending.push({ url, options, resolve, reject }))
};
runInNewContext(source, sandbox);
const wg = sandbox.WG;
wg.toast = (message, type) => toasts.push({ message, type });
const tick = () => new Promise(resolve => setImmediate(resolve));
const reply = async (request, data, ok = true, offline = false) => {
    request.resolve({ ok, headers: { get: () => offline ? '123' : null }, json: async () => data });
    await tick();
};

// Unknown or stale status must not enable a setting write.
wg.refresh();
wg.refresh();
assert.equal(pending.length, 1);
assert.equal(pending[0].options.cache, 'no-store');
await reply(pending.shift(), { expose_dashboard: true }, true, true);
assert.equal(input.disabled, true);
assert.equal(input.indeterminate, true);
wg.changeExposure();
assert.equal(pending.length, 0);
wg.refresh();
await reply(pending.shift(), { expose_dashboard: false });
assert.equal(input.disabled, false);
assert.equal(input.checked, false);
assert.equal(state.textContent, 'Saved setting: off');

// A poll from before a write must not restore the old setting.
wg.refresh();
const stale = pending.shift();
input.checked = true;
wg.changeExposure();
wg.changeExposure();
wg.refresh();
assert.equal(input.disabled, true);
assert.equal(pending.length, 1);
const write = pending.shift();
assert.equal(write.url, '/api/wireguard/expose');
assert.equal(write.options.method, 'POST');
assert.deepEqual(JSON.parse(write.options.body), { enabled: true });
await reply(write, { success: true, expose_dashboard: true });
assert.equal(input.disabled, true);
const readback = pending.shift();
await reply(readback, { expose_dashboard: true });
await reply(stale, { expose_dashboard: false });
assert.equal(input.checked, true);
assert.equal(input.disabled, false);
assert.equal(state.textContent, 'Saved setting: on');
assert.equal(toasts.at(-1).type, 'success');

// A denied step-up write restores the authoritative flag and surfaces the error.
input.checked = false;
wg.changeExposure();
await reply(pending.shift(), { success: false, error: 'Cancelled' }, false);
assert.equal(input.disabled, true);
await reply(pending.shift(), { expose_dashboard: true });
assert.equal(input.checked, true);
assert.equal(input.disabled, false);
assert.equal(toasts.at(-1).message, 'Cancelled');

// Failed readback leaves the control disabled until a later valid poll.
input.checked = false;
wg.changeExposure();
await reply(pending.shift(), { success: true, expose_dashboard: false });
pending.shift().reject(new Error('offline'));
await tick();
assert.equal(input.disabled, true);
assert.equal(input.indeterminate, true);
wg.refresh();
await reply(pending.shift(), { expose_dashboard: false });
assert.equal(input.checked, false);
assert.equal(input.disabled, false);
console.log('WireGuard WebUI exposure checks passed');
