import { View, TouchableOpacity } from "react-native";
import { useLocalSearchParams, router } from "expo-router";
import { Video } from "expo-av";

import { AppText as Text } from "@/components/AppText";

export default function ResultScreen() {
  const { url } = useLocalSearchParams();

  if (!url) {
    return (
      <View className="flex-1 bg-white items-center justify-center">
        <Text>No video found.</Text>
      </View>
    );
  }

  return (
    <View className="flex-1 bg-black">
      {/* Video */}
      <View className="flex-1 justify-center">
        <Video
          source={{ uri: String(url) }}
          style={{
            width: "100%",
            height: 400,
          }}
          useNativeControls
          resizeMode="contain"
          shouldPlay
        />
      </View>

      <Text>Blah blah blah</Text>

      {/* Actions */}
      <View className="px-6 pb-10 pt-4 bg-black">
        <TouchableOpacity
          onPress={() => router.back()}
          className="bg-[#1A3263] py-4 rounded-xl mb-3"
        >
          <Text className="text-white text-center font-outfit-bold">
            Back to Record
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          onPress={() =>
            router.replace("/(tabs)")
          }
          className="border border-white/20 py-4 rounded-xl"
        >
          <Text className="text-white text-center font-outfit-bold">
            Go Home
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}
