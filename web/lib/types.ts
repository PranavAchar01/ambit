export type Verdict = "nominal" | "defect" | "unroutable";

export interface BBoxRegion {
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
}

export interface Candidate {
  model_id: string;
  name: string;
  asset_class: string;
  score: number;
  gate: number;
  training_samples: number | null;
  metrics: ModelMetrics;
}

export interface LayoutModel {
  model_id: string;
  name: string;
  asset_class: string;
  /** illustrative ordering only -- encodes neighbourhood, not distance */
  angle: number;
  gate: number;
  created_by: string | null;
  image_auroc: number | null;
  reference_image_url: string | null;
}

export interface RegistryLayout {
  route_threshold: number;
  angle_is_illustrative: boolean;
  models: LayoutModel[];
}

export interface RecallMatch {
  id: string;
  timestamp: string;
  asset_class: string | null;
  verdict: Verdict;
  severity: number;
  similarity: number;
  routed_model_name: string | null;
  agent_narrative: string;
  uploaded_image_url: string | null;
  heatmap_url: string | null;
}

export interface RecallResponse {
  index: { name: string; status: string | null };
  count: number;
  recurrence: {
    similar_defects: number;
    first_seen: string | null;
    last_seen: string | null;
  };
  matches: RecallMatch[];
}

export interface ModelMetrics {
  image_auroc: number | null;
  pixel_auroc: number | null;
}

export interface ColdStartInfo {
  backbone: string;
  reference_images: number;
  seconds: number;
  image_threshold: number;
  threshold_policy: string;
  model_id: string;
  name: string;
  naming_source: string;
  weights_mb: number;
}

export interface AgentStep {
  node: string;
  message: string;
  at: number;
  [key: string]: unknown;
}

export interface InspectResult {
  verdict: Verdict;
  asset_class: string | null;
  routed_model: { id: string | null; name: string | null };
  routing_score: number;
  anomaly_score: number | null;
  raw_anomaly_score: number | null;
  image_threshold: number | null;
  severity: number;
  bbox_regions: BBoxRegion[];
  candidates: Candidate[];
  decision_source: string | null;
  decision_reason: string | null;
  cold_start: ColdStartInfo | null;
  narrative: string;
  narrative_source: string | null;
  uploaded_image_url: string;
  heatmap_url: string | null;
  reference_image_url: string | null;
  finding_id: string | null;
  latency_ms: number | null;
  thread_id: string | null;
  trace: AgentStep[];
}

export interface RegistryModel {
  id: string;
  name: string;
  asset_class: string;
  backbone: string;
  embedding_dim: number;
  embedding_count: number;
  routing_threshold: number | null;
  image_threshold: number;
  training_samples: number;
  weights_bytes: number;
  metrics: ModelMetrics;
  created_at: string;
  created_by: string;
  reference_image_url: string | null;
}

export interface ModelsResponse {
  count: number;
  models: RegistryModel[];
  vector_index: string | null;
}

export interface TrendPoint {
  day: string;
  asset_class: string;
  total: number;
  defects: number;
  defect_rate: number;
  mean_severity: number;
}

export interface ClassTrend {
  asset_class: string;
  inspections: number;
  defects: number;
  defect_rate: number;
  recent_defect_rate: number;
  mean_severity: number;
  max_severity: number;
  acceleration?: number;
}

export interface TrendsResponse {
  window_days: number;
  generated_at: string;
  defect_rate_over_time: TrendPoint[];
  by_asset_class: ClassTrend[];
  failing_fastest: ClassTrend[];
  severity_distribution: { bucket: string; count: number }[];
  verdict_counts: { verdict: string; count: number }[];
  registry_coverage: {
    total_models: number;
    asset_classes: string[];
    by_origin: { created_by: string; count: number; classes: string[] }[];
    total_inspections: number;
  };
  backfill: {
    count: number;
    total: number;
    fraction: number;
    note: string;
  };
}
