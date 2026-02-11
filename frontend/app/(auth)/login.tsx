import { View, TextInput, TouchableOpacity } from "react-native";
import { useState } from "react";
import { supabase } from "@/lib/supabase";
import { AppText as Text } from "@/components/AppText";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  async function signIn() {
    const { error } =
      await supabase.auth.signInWithPassword({
        email,
        password,
      });

    if (error) alert(error.message);
  }

  return (
    <View className="flex-1 bg-white px-6 justify-center">
      <Text className="text-3xl font-outfit-bold mb-8">
        Welcome Back
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
        onPress={signIn}
        className="bg-[#1A3263] py-4 rounded-xl"
      >
        <Text className="text-white text-center font-outfit-bold">
          Login
        </Text>
      </TouchableOpacity>
    </View>
  );
}
