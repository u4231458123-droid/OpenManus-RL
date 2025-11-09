import "jsr:@supabase/functions-js/edge-runtime.d.ts";

interface AgentStateRequest {
  rollout_id: string;
  step_number: number;
  observation: string;
  action: string;
  thought?: string;
  memory_summary?: string;
  tool_calls?: Array<{
    tool_name: string;
    tool_input: Record<string, any>;
    tool_output?: string;
    success?: boolean;
    error_message?: string;
    execution_time_ms?: number;
  }>;
  reward?: {
    value: number;
    type?: string;
    details?: Record<string, any>;
  };
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, {
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
      },
    });
  }

  try {
    const requestData: AgentStateRequest = await req.json();
    const { rollout_id, step_number, observation, action, thought, memory_summary, tool_calls, reward } = requestData;

    // Validate required fields
    if (!rollout_id || step_number === undefined || !observation || !action) {
      return new Response(
        JSON.stringify({ error: "rollout_id, step_number, observation, and action are required" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

    const { createClient } = await import("https://esm.sh/@supabase/supabase-js@2");
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Insert agent state
    const { data: agentState, error: stateError } = await supabase
      .from("agent_states")
      .insert({
        rollout_id,
        step_number,
        observation,
        action,
        thought,
        memory_summary,
      })
      .select()
      .single();

    if (stateError) {
      return new Response(
        JSON.stringify({ error: "Failed to log agent state", details: stateError.message }),
        { status: 500, headers: { "Content-Type": "application/json" } }
      );
    }

    // Insert tool calls if provided
    if (tool_calls && tool_calls.length > 0) {
      const toolCallData = tool_calls.map(tc => ({
        agent_state_id: agentState.id,
        tool_name: tc.tool_name,
        tool_input: tc.tool_input,
        tool_output: tc.tool_output,
        success: tc.success ?? true,
        error_message: tc.error_message,
        execution_time_ms: tc.execution_time_ms,
      }));

      const { error: toolError } = await supabase
        .from("tool_calls")
        .insert(toolCallData);

      if (toolError) {
        console.error("Failed to log tool calls:", toolError);
      }
    }

    // Insert reward if provided
    if (reward) {
      const { error: rewardError } = await supabase
        .from("rewards")
        .insert({
          rollout_id,
          step_number,
          reward: reward.value,
          reward_type: reward.type || "step",
          details: reward.details || {},
        });

      if (rewardError) {
        console.error("Failed to log reward:", rewardError);
      }
    }

    return new Response(
      JSON.stringify({
        success: true,
        agent_state_id: agentState.id,
        message: "Agent state logged successfully",
      }),
      {
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      }
    );
  } catch (error) {
    console.error("Error:", error);
    return new Response(
      JSON.stringify({ error: "Internal server error", details: error.message }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
});
