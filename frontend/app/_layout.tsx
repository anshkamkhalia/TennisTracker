import "./global.css";

import { Stack, Redirect } from "expo-router";
import { StatusBar } from "expo-status-bar";

import { useFonts } from "expo-font";
import {
  Outfit_400Regular,
  Outfit_500Medium,
  Outfit_600SemiBold,
  Outfit_700Bold,
} from "@expo-google-fonts/outfit";

import * as SplashScreen from "expo-splash-screen";
import { useEffect, useState } from "react";

import { supabase } from "@/lib/supabase";

import {
  MaterialCommunityIcons,
  Ionicons,
} from "@expo/vector-icons";

SplashScreen.preventAutoHideAsync();

export default function RootLayout() {
  /* Fonts */
  const [fontsLoaded] = useFonts({
    Outfit_400Regular,
    Outfit_500Medium,
    Outfit_600SemiBold,
    Outfit_700Bold,

    ...Ionicons.font,
    ...MaterialCommunityIcons.font,
  });

  /* Auth */
  const [session, setSession] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadSession = async () => {
      const { data } = await supabase.auth.getSession();
      setSession(data.session);
      setLoading(false);
    };

    loadSession();

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_e, session) => {
      setSession(session);
    });

    return () => subscription.unsubscribe();
  }, []);

  /* Splash */
  useEffect(() => {
    if (fontsLoaded && !loading) {
      SplashScreen.hideAsync();
    }
  }, [fontsLoaded, loading]);

  if (!fontsLoaded || loading) return null;

  return (
    <>
      {/* 🔐 Auth Guard */}
      {!session && <Redirect href="/(auth)/welcome" />}
      {session && <Redirect href="/(tabs)" />}

      <Stack screenOptions={{ headerShown: false }}>
        <Stack.Screen name="(auth)" />
        <Stack.Screen name="(tabs)" />
        <Stack.Screen
          name="modal"
          options={{ presentation: "modal" }}
        />
      </Stack>

      <StatusBar style="light" />
    </>
  );
}
