// Shapes returned by the FastAPI service (api/main.py, roadcond.export.build_study).

export interface Top2Entry {
  class: string;
  probability: number;
}

export interface PredictResponse {
  prediction: string;
  confidence: number;
  is_uncertain: boolean;
  display_label: string;
  top2: Top2Entry[];
  probabilities: Record<string, number>;
  heatmap_base64: string | null;
  heatmap_media_type: string;
  heatmap_class: string;
  limits_note: string;
  temperature: number;
  uncertainty_threshold: number;
  model: { name: string; mock: boolean };
}

export interface Health {
  status: string;
  version: string;
  model?: string | null;
  classes?: string[] | null;
  mock?: boolean | null;
  detail?: string | null;
  uncertainty_threshold?: number | null;
  calibration?: {
    temperature: number;
    ece_before?: number | null;
    ece_after?: number | null;
    split?: string;
  } | null;
  training?: {
    dataset?: string;
    labels?: string;
    best_epoch?: number | null;
    val_macro_f1?: number | null;
    test_macro_f1?: number | null;
    seed?: number | null;
  } | null;
  provenance?: { checkpoint_sha256?: string | null; onnx_sha256?: string | null } | null;
}

export interface Summary {
  mean: number | null;
  std: number | null;
  n: number;
}

export interface SlopeEntry {
  country: string;
  source_map50: number;
  target_map50: number;
  relative_gap_map50: number | null;
}

export interface CurvePoint {
  labels: number;
  map50: Summary;
}

export interface Study {
  mock: boolean;
  source_country: string;
  source: { map50: number };
  slope: SlopeEntry[];
  learning_curve: Record<string, CurvePoint[]>;
  notes: string[];
}
