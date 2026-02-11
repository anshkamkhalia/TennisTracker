import { View, TouchableOpacity } from "react-native";
import { router } from "expo-router";
import { AppText as Text } from "@/components/AppText";

export default function Welcome() {
  return (
    <View className="flex-1 bg-white items-center justify-center px-6">
      <Text className="text-4xl font-outfit-bold mb-4">
        TennisTracker
      </Text>

      <Text className="text-gray-500 text-center mb-10">
        Improve your game with AI analysis
      </Text>

      <TouchableOpacity
        onPress={() => router.push("/(auth)/login")}
        className="bg-[#1A3263] w-full py-4 rounded-xl mb-4"
      >
        <Text className="text-white text-center font-outfit-bold">
          Login
        </Text>
      </TouchableOpacity>

      <TouchableOpacity
        onPress={() => router.push("/(auth)/signup")}
        className="border border-[#1A3263] w-full py-4 rounded-xl"
      >
        <Text className="text-[#1A3263] text-center font-outfit-bold">
          Sign Up
        </Text>
      </TouchableOpacity>
    </View>
  );
}
