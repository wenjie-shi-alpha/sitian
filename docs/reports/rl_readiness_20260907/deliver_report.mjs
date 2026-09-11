// Use the canonical report builder and verifier, with the project's existing
// narrow fix for the portable reader's 100vw scrollbar overflow.
import { execFileSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { deliverPortableArtifact } from '/root/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.10-13ceeea1f599/skills/build-report/scripts/deliver_portable_artifact.mjs';
import { verifyPortableArtifact } from '/root/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.10-13ceeea1f599/skills/build-report/scripts/verify_portable_artifact.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../../..');
const result = await deliverPortableArtifact({
  inputPath: resolve(here, 'artifact.json'),
  outputPath: resolve(here, 'report.html'),
}, {
  verify: async (options) => {
    execFileSync('python3', [resolve(root, 'scripts/fix_portable_report_layout.py'), options.htmlPath], {stdio: 'pipe'});
    return verifyPortableArtifact(options);
  },
});
writeFileSync(resolve(here, 'delivery_receipt.json'), JSON.stringify(result, null, 2) + '\n');
process.stdout.write(JSON.stringify(result) + '\n');
if (!result.ok) process.exitCode = 1;
