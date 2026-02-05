/**
 * Test GLiNER2 ONNX runtime against pre-generated fixtures.
 *
 * Fixtures were generated from the native PyTorch GLiNER2 implementation
 * and should produce identical results with the ONNX runtime.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it, expect, beforeAll } from 'vitest';
import { GLiNER2ONNXRuntime } from '../src/gliner2/runtime.js';
import type { Precision } from '../src/gliner2/constants.js';
import type { GLiNER2AllFixtures } from './fixtures.types.js';
import * as ort from 'onnxruntime-node';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const USE_GPU = process.env['USE_GPU'] === '1';
const EXECUTION_PROVIDERS: ort.InferenceSession.ExecutionProviderConfig[] = USE_GPU ? ['cuda', 'cpu'] : ['cpu'];
const SCORE_TOLERANCE = 0.05;

const MODEL_IDS: Record<string, string> = {
  'gliner2-multi-v1': 'lmo3/gliner2-multi-v1-onnx',
  'gliner2-large-v1': 'lmo3/gliner2-large-v1-onnx',
};

const PRECISIONS: Precision[] = ['fp32', 'fp16'];

const fixturesPath = path.join(__dirname, 'gliner2.fixtures.json');
const allFixtures: GLiNER2AllFixtures = JSON.parse(fs.readFileSync(fixturesPath, 'utf-8'));

for (const [modelKey, fixtures] of Object.entries(allFixtures)) {
  const modelId = MODEL_IDS[modelKey];

  if (!modelId) {
    describe.skip(`${modelKey} (no model ID configured)`, () => {
      it('skipped', () => {});
    });
    continue;
  }

  for (const precision of PRECISIONS) {
    describe(`${modelKey} (${precision})`, () => {
      let runtime: GLiNER2ONNXRuntime;

      beforeAll(async () => {
        runtime = await GLiNER2ONNXRuntime.fromPretrained(modelId, { precision, executionProviders: EXECUTION_PROVIDERS });
      }, 120000);

      describe('Classification', () => {
        for (const fixture of fixtures.classification) {
          it(`should classify: "${fixture.text.slice(0, 50)}..."`, async () => {
            const result = await runtime.classify(fixture.text, fixture.labels);

            const actualLabel = Object.keys(result)[0] ?? '';
            const actualScore = result[actualLabel] ?? 0;

            expect(actualLabel).toBe(fixture.expected_label);
            expect(Math.abs(actualScore - fixture.expected_score)).toBeLessThanOrEqual(SCORE_TOLERANCE);
          });
        }
      });

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
}
