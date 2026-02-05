export interface EntityFixture {
  text: string;
  label: string;
  score?: number;
  start?: number;
  end?: number;
}

export interface ClassificationFixture {
  text: string;
  labels: string[];
  expected_label: string;
  expected_score: number;
}

export interface NERFixture {
  text: string;
  labels: string[];
  threshold: number;
  expected: EntityFixture[];
}

export interface GLiNER1ModelFixtures {
  ner: NERFixture[];
}

export interface GLiNER2ModelFixtures {
  classification: ClassificationFixture[];
  ner: NERFixture[];
}

export type GLiNER1AllFixtures = Record<string, GLiNER1ModelFixtures>;
export type GLiNER2AllFixtures = Record<string, GLiNER2ModelFixtures>;
