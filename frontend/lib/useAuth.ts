import { useEffect, useState } from "react";
import { supabase } from "./supabase";
import { Session } from "@supabase/supabase-js";

export function useAuth() {
  const [session, setSession] =
    useState<Session | null>(null);

  const [loading, setLoading] = useState(true);

  useEffect(() => {
    supabase.auth
      .getSession()
      .then(({ data }) => {
        setSession(data.session);
        setLoading(false);
      });

    const {
      data: listener,
    } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setSession(session);
      }
    );

    return () => {
      listener.subscription.unsubscribe();
    };
  }, []);

  return { session, loading };
}
