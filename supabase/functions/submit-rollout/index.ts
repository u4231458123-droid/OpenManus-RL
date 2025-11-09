import "jsr:@supabase/functions-js/edge-runtime.d.ts";

interface RolloutRequest {
  training_run_id?: string;
  episode_number: number;
  environment: string;
  task_description?: string;
  metadata?: Record<string, any>;
}

interface RolloutResponse {
  rollout_id: string;
  status: string;
  message: string;
}

Deno.serve(async (req: Request) => {
  // CORS headers
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
    const { training_run_id, episode_number, environment, task_description, metadata }: RolloutRequest =
      await req.json();

    // Validate required fields
    if (!episode_number || !environment) {
      return new Response(
        JSON.stringify({ error: "episode_number and environment are required" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    // Create Supabase client
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

    const { createClient } = await import("https://esm.sh/@supabase/supabase-js@2");
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Insert rollout
    const { data, error } = await supabase
      .from("rollouts")
      .insert({
        training_run_id,
        episode_number,
        environment,
        task_description,
        status: "running",
        metadata: metadata || {},
      })
      .select()
      .single();

    if (error) {
      console.error("Database error:", error);
      return new Response(
        JSON.stringify({ error: "Failed to create rollout", details: error.message }),
        { status: 500, headers: { "Content-Type": "application/json" } }
      );
    }

    const response: RolloutResponse = {
      rollout_id: data.id,
      status: "created",
      message: "Rollout created successfully",
    };

    return new Response(JSON.stringify(response), {
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
