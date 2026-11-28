// Standalone regression harness: node app/src/test/js/network_data_limit_harness.js
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const source = fs.readFileSync(path.join(__dirname, '../../main/assets/web/shared/network.js'), 'utf8');
const input = { value: '', validity: { badInput: false } };
let saved = [], errors = [];
const page = vm.runInNewContext(source + '\nNetworkPage;', {
    document: { getElementById: id => id === 'hsCap' ? input : null },
    BYD: { core: { toast: (message, type) => errors.push(type) } }
});
page.saveSettings = body => saved.push(body.dataCapMb);
for (const value of ['', '0', '00128', '9007199254740991']) {
    input.value = value;
    page.saveLimit();
}
assert.deepStrictEqual(saved, [0, 0, 128, 9007199254740991]);
for (const value of ['-1', '+1', '1.5', '1e3', 'abc', '9007199254740992']) {
    input.value = value;
    page.saveLimit();
}
input.value = '';
input.validity.badInput = true;
page.saveLimit();
assert.strictEqual(saved.length, 4);
assert.strictEqual(errors.length, 7);
console.log('Network data limit: all scenarios passed');
