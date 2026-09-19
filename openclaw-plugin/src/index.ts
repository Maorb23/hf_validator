import { spawn } from "node:child_process";
import { Type } from "typebox";
import { defineToolPlugin } from "openclaw/plugin-sdk/tool-plugin";

const MAX_OUTPUT_BYTES = 2_000_000;

const configSchema = Type.Object(
  {
    projectRoot: Type.String({ minLength: 1 }),
    pythonExecutable: Type.Optional(Type.String({ minLength: 1 })),
    stateRoot: Type.Optional(Type.String({ minLength: 1 })),
    timeoutMs: Type.Optional(Type.Integer({ minimum: 1000, maximum: 600_000 })),
  },
  { additionalProperties: false },
);

const recommendationFields = {
  request: Type.String({ minLength: 1, maxLength: 20_000 }),
  gpu_memory_gb: Type.Optional(Type.Number({ exclusiveMinimum: 0 })),
  gpu_count: Type.Optional(Type.Integer({ minimum: 1, maximum: 64 })),
  budget_usd_per_hour: Type.Optional(Type.Number({ exclusiveMinimum: 0 })),
  gpu_hourly_cost_usd: Type.Optional(Type.Number({ exclusiveMinimum: 0 })),
  license: Type.Optional(Type.String({ minLength: 1, maxLength: 100 })),
  framework_mode: Type.Optional(Type.Union([Type.Literal("fast"), Type.Literal("sweep")])),
};

type PluginConfig = {
  projectRoot: string;
  pythonExecutable?: string;
  stateRoot?: string;
  timeoutMs?: number;
};

function invokeBridge(
  operation: string,
  args: unknown,
  config: PluginConfig,
  signal?: AbortSignal,
): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const child = spawn(
      config.pythonExecutable ?? "python",
      ["-m", "hf_model_watch.openclaw_adapter"],
      {
        cwd: config.projectRoot,
        env: {
          ...process.env,
          HF_MODEL_WATCH_STATE_ROOT: config.stateRoot ?? "artifacts/recommendations",
          PYTHONPATH: `${config.projectRoot}/src${process.env.PYTHONPATH ? `:${process.env.PYTHONPATH}` : ""}`,
        },
        shell: false,
        stdio: ["pipe", "pipe", "pipe"],
      },
    );
    const timeoutMs = config.timeoutMs ?? 300_000;
    const timer = setTimeout(() => child.kill("SIGKILL"), timeoutMs);
    const chunks: Buffer[] = [];
    let size = 0;
    let rejected = false;
    const abort = () => child.kill("SIGTERM");
    signal?.addEventListener("abort", abort, { once: true });
    child.stdout.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_OUTPUT_BYTES) {
        rejected = true;
        child.kill("SIGKILL");
        reject(new Error("hf-model-watch response exceeded the output limit"));
      } else {
        chunks.push(chunk);
      }
    });
    child.on("error", reject);
    child.on("close", () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
      if (rejected) return;
      try {
        const response = JSON.parse(Buffer.concat(chunks).toString("utf8"));
        if (!response.ok) reject(new Error(String(response.error ?? "hf-model-watch failed")));
        else resolve(response.result);
      } catch (error) {
        reject(error instanceof Error ? error : new Error("Malformed bridge response"));
      }
    });
    child.stdin.end(JSON.stringify({ operation, arguments: args }));
  });
}

export default defineToolPlugin({
  id: "hf-model-watch",
  name: "Hugging Face Model Watch",
  description: "Restricted conversational access to hf-model-watch recommendations.",
  configSchema,
  tools: (tool) => [
    tool({
      name: "recommend_models",
      description: "Request authoritative model discovery. Never infer missing results.",
      parameters: Type.Object(recommendationFields, { additionalProperties: false }),
      execute: (args, config, context) => invokeBridge("recommend_models", args, config, context.signal),
    }),
    tool({
      name: "provide_recommendation_input",
      description: "Answer only the outstanding structured questions for an existing run.",
      parameters: Type.Object(
        {
          run_id: Type.String({ pattern: "^run_[A-Za-z0-9_-]+$" }),
          answers: Type.Record(Type.String(), Type.Union([Type.String(), Type.Number()])),
        },
        { additionalProperties: false },
      ),
      execute: (args, config, context) => invokeBridge("provide_recommendation_input", args, config, context.signal),
    }),
    tool({
      name: "select_model",
      description: "Persist a model choice only after the user explicitly confirms it.",
      parameters: Type.Object(
        {
          run_id: Type.String({ pattern: "^run_[A-Za-z0-9_-]+$" }),
          model_id: Type.String({ minLength: 1, maxLength: 500 }),
          confirmed: Type.Literal(true),
        },
        { additionalProperties: false },
      ),
      execute: (args, config, context) => invokeBridge("select_model", args, config, context.signal),
    }),
    tool({
      name: "record_model_feedback",
      description: "Record user-reported feedback for the selected model; it remains unverified.",
      parameters: Type.Object(
        {
          run_id: Type.String({ pattern: "^run_[A-Za-z0-9_-]+$" }),
          model_id: Type.String({ minLength: 1, maxLength: 500 }),
          category: Type.Union([
            Type.Literal("out_of_memory"), Type.Literal("latency"),
            Type.Literal("throughput"), Type.Literal("quality"),
            Type.Literal("capability"), Type.Literal("other"),
          ]),
          details: Type.String({ minLength: 1, maxLength: 10_000 }),
          context: Type.Optional(Type.Record(Type.String(), Type.Union([
            Type.String(), Type.Number(), Type.Boolean(), Type.Null(),
          ]))),
        },
        { additionalProperties: false },
      ),
      execute: (args, config, context) => invokeBridge("record_model_feedback", args, config, context.signal),
    }),
    tool({
      name: "revise_recommendation",
      description: "Create a bounded revised discovery linked to its parent run.",
      parameters: Type.Object(
        {
          ...recommendationFields,
          parent_run_id: Type.String({ pattern: "^run_[A-Za-z0-9_-]+$" }),
          revision_reason: Type.String({ minLength: 1, maxLength: 1000 }),
          trigger_feedback_id: Type.Optional(Type.String({ pattern: "^fb_[0-9a-f]{32}$" })),
        },
        { additionalProperties: false },
      ),
      execute: (args, config, context) => invokeBridge("revise_recommendation", args, config, context.signal),
    }),
    tool({
      name: "get_run_status",
      description: "Retrieve canonical persisted state for an existing recommendation run.",
      parameters: Type.Object(
        { run_id: Type.String({ pattern: "^run_[A-Za-z0-9_-]+$" }) },
        { additionalProperties: false },
      ),
      execute: (args, config, context) => invokeBridge("get_run_status", args, config, context.signal),
    }),
  ],
});
