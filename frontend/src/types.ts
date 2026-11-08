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
  review?: Review | null; // null when review routing is off on the server
}

// Review routing (roadcond.routing.live): how much human review a photo gets.
export type ReviewMode = "light_review" | "full_review" | "second_review";

export interface Review {
  decision_id: string | null; // null only when the server could not record the decision
  mode: ReviewMode;
  reasons: string[];
  confidence: number | null; // the review rule's temperature-scaled confidence
  fallback: boolean;
  reviews_needed: number;
  recorded: boolean;
}

export interface ReviewItem {
  decision_id: string;
  routed_at: string;
  prediction: string | null;
  mode: ReviewMode;
  reasons: string[];
  confidence: number | null;
  status: "pending" | "disputed" | "done";
  agreed_label: string | null;
  reviews_needed: number;
  reviewers: string[];
  has_image: boolean;
}

export interface ReviewQueue {
  reviewer: string;
  counts: Record<ReviewMode, { open: number; done: number }>;
  items: ReviewItem[];
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
  routing?: {
    enabled: boolean;
    detail?: string | null;
    reviewing?: boolean;
    stores_images?: boolean;
    thresholds?: { high: number; low: number };
    safety_critical_classes?: string[];
  } | null;
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
