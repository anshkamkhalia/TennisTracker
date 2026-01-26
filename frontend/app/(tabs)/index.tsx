import { View, ScrollView, TouchableOpacity } from 'react-native';
import { AppText as Text } from "@/components/AppText";
import { MaterialCommunityIcons, Ionicons } from '@expo/vector-icons';

export default function HomeScreen() {
  return (
    <ScrollView className="flex-1 bg-white font-sans"  >
      {/* Header */}
      <View className="pt-16 pb-6 px-6">
        <View className="flex-row items-center justify-between mb-1">
          <View>
            <Text className="text-gray-900 text-4xl" style={{fontFamily: "Outfit_700Bold"}}>TennisTracker</Text>
            <Text className="text-gray-500 text-base mt-1">Track your performance</Text>
          </View>
          <TouchableOpacity>
            <View className="w-14 h-14 rounded-full overflow-hidden" style={{ backgroundColor: '#1A3263' }}>
              <View className="w-full h-full items-center justify-center">
                <Text className="text-white text-xl" style={{fontFamily: "Outfit_700Bold"}}>AP</Text>
              </View>
            </View>
          </TouchableOpacity>
        </View>
      </View>

      {/* Main Action Cards */}
      <View className="px-6 mb-6">
        <View className="flex-row gap-3 mb-3">
          {/* Analyze Video - Large Card */}
          <TouchableOpacity className="flex-1 rounded-[28px] p-6 h-64" style={{ backgroundColor: '#E8E3FF' }}>
            <View className="flex-1 justify-between">
              <View className="bg-white w-14 h-14 rounded-full items-center justify-center">
                <Ionicons name="videocam" size={28} style={{ color: '#1A3263' }} />
              </View>
              <View>
                <Text className="text-2xl font-bold mb-1" style={{ color: '#1A3263' }}>
                  Analyze {'\n'}Video
                </Text>
                <Text className="text-sm" style={{ color: '#1A3263', opacity: 0.6 }}>
                  Get instant feedback
                </Text>
              </View>
            </View>
          </TouchableOpacity>

          {/* Right Column */}
          <View className="flex-1 gap-3">
            {/* History Card */}
            <TouchableOpacity className="rounded-[28px] p-5 flex-1 justify-between" style={{ backgroundColor: '#FFF4E6' }}>
              <View className="bg-white w-12 h-12 rounded-full items-center justify-center">
                <Ionicons name="time" size={24} style={{ color: '#1A3263' }} />
              </View>
              <View>
                <Text className="text-lg font-bold" style={{ color: '#1A3263' }}>History</Text>
                <Text className="text-xs mt-0.5" style={{ color: '#1A3263', opacity: 0.6 }}>24 sessions</Text>
              </View>
            </TouchableOpacity>

            {/* Stats Card */}
            <TouchableOpacity className="rounded-[28px] p-5 flex-1 justify-between" style={{ backgroundColor: '#1A3263' }}>
              <View className="bg-white/20 w-12 h-12 rounded-full items-center justify-center">
                <MaterialCommunityIcons name="chart-line" size={24} color="white" />
              </View>
              <View>
                <Text className="text-white text-lg font-bold">Stats</Text>
                <Text className="text-white/60 text-xs mt-0.5">View insights</Text>
              </View>
            </TouchableOpacity>
          </View>
        </View>
      </View>

      {/* Performance Overview */}
      <View className="px-6 mb-6">
        <Text className="text-gray-900 text-xl mb-4" style={{fontFamily: "Outfit_600SemiBold"}}>Performance</Text>
        
        <View className="flex-row gap-3">
          <View className="flex-1 bg-white rounded-3xl p-5 border border-gray-100">
            <View className="bg-gray-50 w-12 h-12 rounded-2xl items-center justify-center mb-3">
              <MaterialCommunityIcons name="target" size={24} style={{ color: '#1A3263' }} />
            </View>
            <Text className="text-3xl font-bold mb-1" style={{ color: '#1A3263' }}>2,847</Text>
            <Text className="text-gray-500 text-sm">Total Shots</Text>
            <View className="mt-2 bg-emerald-50 self-start px-2 py-1 rounded-full">
              <Text className="text-emerald-600 text-xs font-bold">+12%</Text>
            </View>
          </View>

          <View className="flex-1 bg-white rounded-3xl p-5 border border-gray-100">
            <View className="bg-gray-50 w-12 h-12 rounded-2xl items-center justify-center mb-3">
              <Ionicons name="flash" size={24} style={{ color: '#1A3263' }} />
            </View>
            <Text className="text-3xl font-bold mb-1" style={{ color: '#1A3263' }}>94</Text>
            <Text className="text-gray-500 text-sm">mph Avg</Text>
            <View className="mt-2 bg-amber-50 self-start px-2 py-1 rounded-full">
              <Text className="text-amber-600 text-xs font-bold">+5%</Text>
            </View>
          </View>
        </View>

        <View className="flex-row gap-3 mt-3">
          <View className="flex-1 bg-white rounded-3xl p-5 border border-gray-100">
            <View className="bg-gray-50 w-12 h-12 rounded-2xl items-center justify-center mb-3">
              <MaterialCommunityIcons name="trophy" size={24} style={{ color: '#1A3263' }} />
            </View>
            <Text className="text-3xl font-bold mb-1" style={{ color: '#1A3263' }}>342</Text>
            <Text className="text-gray-500 text-sm">Winners</Text>
            <View className="mt-2 bg-blue-50 self-start px-2 py-1 rounded-full">
              <Text className="text-blue-600 text-xs font-bold">+18%</Text>
            </View>
          </View>

          <View className="flex-1 bg-white rounded-3xl p-5 border border-gray-100">
            <View className="bg-gray-50 w-12 h-12 rounded-2xl items-center justify-center mb-3">
              <MaterialCommunityIcons name="bullseye-arrow" size={24} style={{ color: '#1A3263' }} />
            </View>
            <Text className="text-3xl font-bold mb-1" style={{ color: '#1A3263' }}>76%</Text>
            <Text className="text-gray-500 text-sm">Accuracy</Text>
            <View className="mt-2 bg-purple-50 self-start px-2 py-1 rounded-full">
              <Text className="text-purple-600 text-xs font-bold">+3%</Text>
            </View>
          </View>
        </View>
      </View>

      {/* Recent Activity */}
      <View className="px-6 pb-24">
        <View className="flex-row justify-between items-center mb-4">
          <Text className="text-gray-900 text-xl font-bold">Recent Activity</Text>
          <TouchableOpacity>
            <Text className="text-sm font-semibold" style={{ color: '#1A3263' }}>See All</Text>
          </TouchableOpacity>
        </View>

        {/* Activity Item 1 */}
        <TouchableOpacity className="bg-white rounded-3xl p-4 mb-3 flex-row items-center border border-gray-100">
          <View className="w-14 h-14 rounded-2xl items-center justify-center mr-4" style={{ backgroundColor: '#E8E3FF' }}>
            <MaterialCommunityIcons name="tennis" size={28} style={{ color: '#1A3263' }} />
          </View>
          <View className="flex-1">
            <Text className="text-base font-bold mb-1" style={{ color: '#1A3263' }}>
              Morning Practice Session
            </Text>
            <Text className="text-xs text-gray-500 font-medium">Jan 24 • 45:32</Text>
          </View>
          <Ionicons name="ellipsis-horizontal" size={20} color="#9ca3af" />
        </TouchableOpacity>

        {/* Activity Item 2 */}
        <TouchableOpacity className="bg-white rounded-3xl p-4 mb-3 flex-row items-center border border-gray-100">
          <View className="w-14 h-14 rounded-2xl items-center justify-center mr-4" style={{ backgroundColor: '#FFF4E6' }}>
            <MaterialCommunityIcons name="tennis-ball" size={28} style={{ color: '#1A3263' }} />
          </View>
          <View className="flex-1">
            <Text className="text-base font-bold mb-1" style={{ color: '#1A3263' }}>
              Match vs. John
            </Text>
            <Text className="text-xs text-gray-500 font-medium">Jan 22 • 1:23:15</Text>
          </View>
          <Ionicons name="ellipsis-horizontal" size={20} color="#9ca3af" />
        </TouchableOpacity>

        {/* Activity Item 3 */}
        <TouchableOpacity className="bg-white rounded-3xl p-4 mb-3 flex-row items-center border border-gray-100">
          <View className="w-14 h-14 rounded-2xl items-center justify-center mr-4" style={{ backgroundColor: '#E0F2FE' }}>
            <MaterialCommunityIcons name="racquetball" size={28} style={{ color: '#1A3263' }} />
          </View>
          <View className="flex-1">
            <Text className="text-base font-bold mb-1" style={{ color: '#1A3263' }}>
              Serve Practice
            </Text>
            <Text className="text-xs text-gray-500 font-medium">Jan 20 • 28:45</Text>
          </View>
          <Ionicons name="ellipsis-horizontal" size={20} color="#9ca3af" />
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}