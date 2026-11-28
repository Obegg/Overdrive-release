// Standalone regression harness: node app/src/test/js/wireguard_exposure_harness.js
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const source = fs.readFileSync(path.join(__dirname, '../../main/assets/web/shared/wireguard.js'), 'utf8');
const input = { checked: false, disabled: true };
const pending = [];
const toasts = [];
const page = vm.runInNewContext(source + '\nWG;', {
    document: { getElementById: id => id === 'wgExpose' ? input : null },
    window: {},
    fetch: (url, options) => new Promise(resolve => pending.push({ url, options, resolve }))
});
page.toast = (message, type) => toasts.push(type);
const reply = (request, data, ok = true) => request.resolve({ ok, json: async () => data });
const tick = () => new Promise(resolve => setImmediate(resolve));

async function main() {
    page.render({ expose_dashboard: false });
    assert.strictEqual(input.disabled, false);
    const stale = page.refresh();
    const oldGet = pending.shift();
    page.refresh();
    assert.strictEqual(pending.length, 0);
    input.checked = true;
    const write = page.setExpose(input);
    const post = pending.shift();
    assert.strictEqual(post.url, '/api/wireguard/expose');
    assert.strictEqual(post.options.method, 'POST');
    assert.deepStrictEqual(JSON.parse(post.options.body), { enabled: true });
    assert.strictEqual(input.disabled, true);
    page.refresh();
    assert.strictEqual(pending.length, 0);
    reply(oldGet, { expose_dashboard: false });
    await stale;
    assert.strictEqual(input.checked, true);
    reply(post, { success: true, expose_dashboard: true });
    await tick();
    assert.strictEqual(input.disabled, true);
    reply(pending.shift(), { expose_dashboard: true });
    await write;
    assert.strictEqual(input.checked, true);
    assert.strictEqual(input.disabled, false);
    assert.deepStrictEqual(toasts, ['success']);

    input.checked = false;
    const failed = page.setExpose(input);
    reply(pending.shift(), { success: false, error: 'Cancelled' }, false);
    await tick();
    assert.strictEqual(input.checked, true);
    reply(pending.shift(), { expose_dashboard: true });
    await failed;
    assert.deepStrictEqual(toasts, ['success', 'error']);
    assert.strictEqual(input.checked, true);
    assert.strictEqual(input.disabled, false);
    page.render(null);
    assert.strictEqual(input.disabled, true);
    page.render({});
    assert.strictEqual(input.disabled, true);
    console.log('WireGuard exposure: all scenarios passed');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
