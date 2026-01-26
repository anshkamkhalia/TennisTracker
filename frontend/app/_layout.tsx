import "./global.css";
import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { useFonts } from "expo-font";
import {
  Outfit_400Regular,
  Outfit_500Medium,
  Outfit_600SemiBold,
  Outfit_700Bold,
} from "@expo-google-fonts/outfit";
import * as SplashScreen from "expo-splash-screen";
import { useEffect } from "react";
import { Text } from "react-native";
import { MaterialCommunityIcons, Ionicons } from '@expo/vector-icons';

SplashScreen.preventAutoHideAsync();

export const unstable_settings = {
  anchor: "(tabs)",
};

export default function RootLayout() {

  // Loads in fonts and icons
  const [fontsLoaded] = useFonts({
    Outfit_400Regular,
    Outfit_500Medium,
    Outfit_600SemiBold,
    Outfit_700Bold,

    ...Ionicons.font,
    ...MaterialCommunityIcons.font,
  });

  // useEffect(() => {
  //   if (fontsLoaded) {
  //     (Text as any).defaultProps = (Text as any).defaultProps || {};
  //     (Text as any).defaultProps.style = {
  //       fontFamily: "Outfit_400Regular",
  //     };

  //     SplashScreen.hideAsync();
  //   }
  // }, [fontsLoaded]);

  // if (!fontsLoaded) return null;

  return (
    <>
      <Stack>
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen
          name="modal"
          options={{ presentation: "modal", title: "Modal" }}
        />
      </Stack>
      <StatusBar style="light" />
    </>
  );
}
