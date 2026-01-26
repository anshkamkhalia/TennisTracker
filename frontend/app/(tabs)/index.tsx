import { View, Text, ScrollView, TouchableOpacity } from 'react-native';
import { MaterialCommunityIcons, Ionicons } from '@expo/vector-icons';
import { Link } from 'expo-router';

export default function HomeScreen() {
  return (
    <ScrollView className="flex-1 bg-green-500 outfit">
      {/* Header */}
      <View className="pt-12 pb-6 px-6">
        <Text className="text-white text-3xl font-bold">Tennis Analyzer</Text>
        <Text className="text-white text-base opacity-90">Track, Analyze, Improve</Text>
      </View>

      {/* Stats Grid */}
      <View className="px-6 pb-6">
        <View className="flex-row gap-4 mb-4">
          {/* Total Shots Card */}
          <View className="flex-1 bg-white rounded-2xl p-5">
            <View className="flex-row justify-between items-start mb-3">
              <View className="bg-green-100 p-2 rounded-full">
                <MaterialCommunityIcons name="target" size={24} color="#22c55e" />
              </View>
              <Text className="text-green-500 font-semibold">+12%</Text>
            </View>
            <Text className="text-3xl font-bold text-gray-900 mb-1">2,847</Text>
            <Text className="text-gray-500 text-sm">Total Shots</Text>
          </View>

          {/* Avg Speed Card */}
          <View className="flex-1 bg-white rounded-2xl p-5">
            <View className="flex-row justify-between items-start mb-3">
              <View className="bg-green-100 p-2 rounded-full">
                <Ionicons name="flash" size={24} color="#22c55e" />
              </View>
              <Text className="text-green-500 font-semibold">+5%</Text>
            </View>
            <Text className="text-3xl font-bold text-gray-900 mb-1">94 mph</Text>
            <Text className="text-gray-500 text-sm">Avg Speed</Text>
          </View>
        </View>

        <View className="flex-row gap-4">
          {/* Winners Card */}
          <View className="flex-1 bg-white rounded-2xl p-5">
            <View className="flex-row justify-between items-start mb-3">
              <View className="bg-green-100 p-2 rounded-full">
                <MaterialCommunityIcons name="trophy" size={24} color="#22c55e" />
              </View>
              <Text className="text-green-500 font-semibold">+18%</Text>
            </View>
            <Text className="text-3xl font-bold text-gray-900 mb-1">342</Text>
            <Text className="text-gray-500 text-sm">Winners</Text>
          </View>

          {/* Accuracy Card */}
          <View className="flex-1 bg-white rounded-2xl p-5">
            <View className="flex-row justify-between items-start mb-3">
              <View className="bg-green-100 p-2 rounded-full">
                <MaterialCommunityIcons name="chart-line" size={24} color="#22c55e" />
              </View>
              <Text className="text-green-500 font-semibold">+3%</Text>
            </View>
            <Text className="text-3xl font-bold text-gray-900 mb-1">76%</Text>
            <Text className="text-gray-500 text-sm">Accuracy</Text>
          </View>
        </View>
      </View>

      {/* Analyze Button */}
      <View className="px-6 pb-6">
        <TouchableOpacity className="bg-green-600 rounded-2xl py-4 flex-row justify-center items-center">
          <Ionicons name="play" size={24} color="white" />
          <Text className="text-white text-lg font-semibold ml-2">Analyze New Video</Text>
        </TouchableOpacity>
      </View>

      {/* Recent Analyses */}
      <View className="bg-white rounded-t-3xl pt-6 px-6 pb-20">
        <View className="flex-row justify-between items-center mb-4">
          <Text className="text-xl font-bold text-gray-900">Recent Analyses</Text>
          <Text className="text-green-500 font-semibold">View All</Text>
        </View>

        {/* Analysis Item 1 */}
        <TouchableOpacity className="flex-row bg-gray-50 rounded-2xl p-4 mb-3">
          <View className="bg-gray-200 rounded-xl w-16 h-16 justify-center items-center">
            <MaterialCommunityIcons name="image-outline" size={32} color="#9ca3af" />
          </View>
          <View className="flex-1 ml-4">
            <Text className="text-base font-semibold text-gray-900 mb-1">
              Morning Practice Session
            </Text>
            <View className="flex-row items-center gap-3">
              <View className="flex-row items-center">
                <Ionicons name="time-outline" size={14} color="#6b7280" />
                <Text className="text-xs text-gray-500 ml-1">2026-01-24 • 45:32</Text>
              </View>
            </View>
            <View className="flex-row items-center gap-3 mt-1">
              <View className="flex-row items-center">
                <MaterialCommunityIcons name="target" size={14} color="#6b7280" />
                <Text className="text-xs text-gray-500 ml-1">156</Text>
              </View>
              <View className="flex-row items-center">
                <Ionicons name="arrow-up" size={14} color="#6b7280" />
                <Text className="text-xs text-gray-500 ml-1">12</Text>
              </View>
              <View className="flex-row items-center">
                <MaterialCommunityIcons name="tennis" size={14} color="#6b7280" />
                <Text className="text-xs text-gray-500 ml-1">34</Text>
              </View>
            </View>
          </View>
        </TouchableOpacity>

        {/* Analysis Item 2 */}
        <TouchableOpacity className="flex-row bg-gray-50 rounded-2xl p-4 mb-3">
          <View className="bg-gray-200 rounded-xl w-16 h-16 justify-center items-center">
            <MaterialCommunityIcons name="image-outline" size={32} color="#9ca3af" />
          </View>
          <View className="flex-1 ml-4">
            <Text className="text-base font-semibold text-gray-900 mb-1">Match vs. John</Text>
            <View className="flex-row items-center gap-3">
              <View className="flex-row items-center">
                <Ionicons name="time-outline" size={14} color="#6b7280" />
                <Text className="text-xs text-gray-500 ml-1">2026-01-22 • 1:23:15</Text>
              </View>
            </View>
            <View className="flex-row items-center gap-3 mt-1">
              <View className="flex-row items-center">
                <MaterialCommunityIcons name="target" size={14} color="#6b7280" />
                <Text className="text-xs text-gray-500 ml-1">287</Text>
              </View>
              <View className="flex-row items-center">
                <Ionicons name="arrow-up" size={14} color="#6b7280" />
                <Text className="text-xs text-gray-500 ml-1">8</Text>
              </View>
              <View className="flex-row items-center">
                <MaterialCommunityIcons name="tennis" size={14} color="#6b7280" />
                <Text className="text-xs text-gray-500 ml-1">42</Text>
              </View>
            </View>
          </View>
        </TouchableOpacity>

        {/* Analysis Item 3 */}
        <TouchableOpacity className="flex-row bg-gray-50 rounded-2xl p-4 mb-3">
          <View className="bg-gray-200 rounded-xl w-16 h-16 justify-center items-center">
            <MaterialCommunityIcons name="image-outline" size={32} color="#9ca3af" />
          </View>
          <View className="flex-1 ml-4">
            <Text className="text-base font-semibold text-gray-900 mb-1">Serve Practice</Text>
            <View className="flex-row items-center gap-3">
              <View className="flex-row items-center">
                <Ionicons name="time-outline" size={14} color="#6b7280" />
                <Text className="text-xs text-gray-500 ml-1">2026-01-20 • 28:45</Text>
              </View>
            </View>
          </View>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}