import "jsr:@supabase/functions-js/edge-runtime.d.ts";

interface GetMetricsRequest {
  training_run_id?: string;
  environment?: string;
  limit?: number;
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, {
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
      },
    });
  }

  try {
    const url = new URL(req.url);
    const training_run_id = url.searchParams.get("training_run_id");
    const environment = url.searchParams.get("environment");
    const limit = parseInt(url.searchParams.get("limit") || "100");

    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

    const { createClient } = await import("https://esm.sh/@supabase/supabase-js@2");
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Build query
    let query = supabase
      .from("rollouts")
      .select(`
        id,
        episode_number,
        environment,
        status,
        total_reward,
        step_count,
        started_at,
        completed_at,
        training_run_id
      `)
      .order("started_at", { ascending: false })
      .limit(limit);

    if (training_run_id) {
      query = query.eq("training_run_id", training_run_id);
    }

    if (environment) {
      query = query.eq("environment", environment);
    }

    const { data: rollouts, error } = await query;

    if (error) {
      return new Response(
        JSON.stringify({ error: "Failed to fetch metrics", details: error.message }),
        { status: 500, headers: { "Content-Type": "application/json" } }
      );
    }

    // Calculate statistics
    const completed = rollouts.filter(r => r.status !== "running");
    const successful = rollouts.filter(r => r.status === "success");

    const avgReward = completed.length > 0
      ? completed.reduce((sum, r) => sum + (r.total_reward || 0), 0) / completed.length
      : 0;

    const avgSteps = completed.length > 0
      ? completed.reduce((sum, r) => sum + (r.step_count || 0), 0) / completed.length
      : 0;

    const successRate = completed.length > 0
      ? successful.length / completed.length
      : 0;

    const metrics = {
      total_rollouts: rollouts.length,
      completed_rollouts: completed.length,
      successful_rollouts: successful.length,
      success_rate: successRate,
      average_reward: avgReward,
      average_steps: avgSteps,
      rollouts: rollouts.slice(0, 20), // Return last 20 rollouts
    };

    return new Response(JSON.stringify(metrics), {
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
    });
  } catch (error) {
    console.error("Error:", error);
    return new Response(
      JSON.stringify({ error: "Internal server error", details: error.message }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
});
