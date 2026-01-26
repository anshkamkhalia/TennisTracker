import { View, TextInput, TouchableOpacity } from "react-native";
import { useState } from "react";
import { supabase } from "@/lib/supabase";
import { AppText as Text } from "@/components/AppText";

export default function Signup() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  async function signUp() {
    const { error } =
      await supabase.auth.signUp({
        email,
        password,
      });

    if (error) alert(error.message);
  }

  return (
    <View className="flex-1 bg-white px-6 justify-center">
      <Text className="text-3xl font-outfit-bold mb-8">
        Create Account
      </Text>

      <TextInput
        placeholder="Email"
        autoCapitalize="none"
        className="border border-gray-200 rounded-xl p-4 mb-4"
        value={email}
        onChangeText={setEmail}
      />

      <TextInput
        placeholder="Password"
        secureTextEntry
        className="border border-gray-200 rounded-xl p-4 mb-6"
        value={password}
        onChangeText={setPassword}
      />

      <TouchableOpacity
        onPress={signUp}
        className="bg-[#1A3263] py-4 rounded-xl"
      >
        <Text className="text-white text-center font-outfit-bold">
          Sign Up
        </Text>
      </TouchableOpacity>
    </View>
  );
}
