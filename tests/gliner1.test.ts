/**
 * Test GLiNER1 ONNX runtime against pre-generated fixtures.
 *
 * Fixtures were generated from the native PyTorch GLiNER1 implementation
 * and should produce identical results with the ONNX runtime.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it, expect, beforeAll } from 'vitest';
import { GLiNER1ONNXRuntime } from '../src/gliner1/runtime.js';
import type { GLiNER1AllFixtures } from './fixtures.types.js';
import * as ort from 'onnxruntime-node';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const USE_GPU = process.env['USE_GPU'] === '1';
const EXECUTION_PROVIDERS: ort.InferenceSession.ExecutionProviderConfig[] = USE_GPU ? ['cuda', 'cpu'] : ['cpu'];

const MODEL_IDS: Record<string, string> = {
  'gliner_small-v2.1': 'onnx-community/gliner_small-v2.1',
  'gliner_multi-v2.1': 'onnx-community/gliner_multi-v2.1',
  'gliner_large-v2.1': 'onnx-community/gliner_large-v2.1',
};

const fixturesPath = path.join(__dirname, 'gliner1.fixtures.json');
const allFixtures: GLiNER1AllFixtures = JSON.parse(fs.readFileSync(fixturesPath, 'utf-8'));

for (const [modelKey, fixtures] of Object.entries(allFixtures)) {
  const modelId = MODEL_IDS[modelKey];

  if (!modelId) {
    describe.skip(`${modelKey} (no model ID configured)`, () => {
      it('skipped', () => {});
    });
    continue;
  }

  describe(`${modelKey}`, () => {
    let runtime: GLiNER1ONNXRuntime;

    beforeAll(async () => {
      runtime = await GLiNER1ONNXRuntime.fromPretrained(modelId, undefined, {
        executionProviders: EXECUTION_PROVIDERS,
      });
    }, 120000);

    describe('NER', () => {
      for (const fixture of fixtures.ner) {
        it(`should extract entities from: "${fixture.text.slice(0, 50)}..."`, async () => {
          const entities = await runtime.extractEntities(fixture.text, fixture.labels, {
            threshold: fixture.threshold,
          });

          // Convert to comparable format (text, label) pairs
          const actualSet = new Set(entities.map((e) => `${e.text}|${e.label}`));
          const expectedSet = new Set(fixture.expected.map((e) => `${e.text}|${e.label}`));

          // Check that sets match
          const missing = [...expectedSet].filter((v) => !actualSet.has(v));
          const extra = [...actualSet].filter((v) => !expectedSet.has(v));

          if (missing.length > 0 || extra.length > 0) {
            const errorMsg = [
              `Entity mismatch for "${fixture.text}"`,
              missing.length > 0 ? `Missing: ${JSON.stringify(missing)}` : '',
              extra.length > 0 ? `Extra: ${JSON.stringify(extra)}` : '',
            ]
              .filter(Boolean)
              .join('\n');
            expect.fail(errorMsg);
          }
        });
      }
    });
  });
}
