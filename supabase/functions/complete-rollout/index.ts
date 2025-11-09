import "jsr:@supabase/functions-js/edge-runtime.d.ts";

interface CompleteRolloutRequest {
  rollout_id: string;
  status: "success" | "failed" | "timeout";
  total_reward: number;
  step_count: number;
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
    const { rollout_id, status, total_reward, step_count }: CompleteRolloutRequest = await req.json();

    if (!rollout_id || !status || total_reward === undefined || step_count === undefined) {
      return new Response(
        JSON.stringify({ error: "rollout_id, status, total_reward, and step_count are required" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

    const { createClient } = await import("https://esm.sh/@supabase/supabase-js@2");
    const supabase = createClient(supabaseUrl, supabaseKey);

    const { data, error } = await supabase
      .from("rollouts")
      .update({
        status,
        total_reward,
        step_count,
        completed_at: new Date().toISOString(),
      })
      .eq("id", rollout_id)
      .select()
      .single();

    if (error) {
      return new Response(
        JSON.stringify({ error: "Failed to complete rollout", details: error.message }),
        { status: 500, headers: { "Content-Type": "application/json" } }
      );
    }

    return new Response(
      JSON.stringify({
        success: true,
        rollout: data,
        message: "Rollout completed successfully",
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
