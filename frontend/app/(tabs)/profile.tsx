import { View, ScrollView, TouchableOpacity, Image } from "react-native";
import { AppText as Text } from "@/components/AppText";
import { Ionicons, MaterialCommunityIcons } from "@expo/vector-icons";

export default function ProfileScreen() {
  return (
    <ScrollView className="flex-1 bg-white">
      {/* Header */}
      <View className="pt-16 pb-6 px-6 items-center">
        <View className="relative">
          <Image
            source={{ uri: "https://i.pravatar.cc/200" }}
            className="w-28 h-28 rounded-full"
          />

          <TouchableOpacity className="absolute bottom-0 right-0 bg-[#1A3263] p-2 rounded-full">
            <Ionicons name="camera" size={18} color="white" />
          </TouchableOpacity>
        </View>

        <Text className="text-2xl font-outfit-bold mt-4 text-gray-900">
          Alex Parker
        </Text>

        <Text className="text-gray-500 mt-1">
          Intermediate Player
        </Text>
      </View>

      {/* Stats */}
      <View className="flex-row justify-around px-6 mb-6">
        {[
          { label: "Sessions", value: "124" },
          { label: "Wins", value: "86" },
          { label: "Rank", value: "#14" },
        ].map((item) => (
          <View key={item.label} className="items-center">
            <Text className="text-xl font-outfit-bold text-[#1A3263]">
              {item.value}
            </Text>
            <Text className="text-gray-500 text-xs">
              {item.label}
            </Text>
          </View>
        ))}
      </View>

      {/* Menu */}
      <View className="px-6 space-y-3">
        <ProfileItem icon="person" label="Edit Profile" />
        <ProfileItem icon="settings" label="Settings" />
        <ProfileItem icon="shield-checkmark" label="Privacy" />
        <ProfileItem icon="log-out-outline" label="Logout" danger />
      </View>
    </ScrollView>
  );
}

function ProfileItem({
  icon,
  label,
  danger,
}: {
  icon: any;
  label: string;
  danger?: boolean;
}) {
  return (
    <TouchableOpacity className="flex-row items-center bg-white border border-gray-100 rounded-2xl p-4">
      <View
        className={`w-10 h-10 rounded-full items-center justify-center ${
          danger ? "bg-red-50" : "bg-gray-50"
        }`}
      >
        <Ionicons
          name={icon}
          size={20}
          color={danger ? "#dc2626" : "#1A3263"}
        />
      </View>

      <Text
        className={`flex-1 ml-3 ${
          danger ? "text-red-600" : "text-gray-800"
        }`}
      >
        {label}
      </Text>

      <Ionicons
        name="chevron-forward"
        size={18}
        color="#9ca3af"
      />
    </TouchableOpacity>
  );
}
