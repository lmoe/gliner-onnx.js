/**
 * Simple test script to verify ONNX runtime works.
 */

import { GLiNER2ONNXRuntime } from '../src/gliner2/runtime.js';

const MODEL_ID = 'lmo3/gliner2-multi-v1-onnx';

async function main() {
  console.log('Loading model...');
  const runtime = await GLiNER2ONNXRuntime.fromPretrained(MODEL_ID);
  console.log('Model loaded');

  console.log('Testing classification...');
  const result = await runtime.classify('This movie was fantastic!', ['positive', 'negative', 'neutral']);
  console.log('Classification result:', result);

  console.log('Testing NER...');
  const entities = await runtime.extractEntities('John works at Google in Seattle', ['person', 'organization', 'location']);
  console.log('NER result:', entities);
}

main().catch(console.error);
