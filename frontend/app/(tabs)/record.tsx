import { View, TouchableOpacity, ActivityIndicator } from "react-native";
import { useEffect, useRef, useState } from "react";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as ImagePicker from "expo-image-picker";

import { AppText as Text } from "@/components/AppText";
import { Ionicons } from "@expo/vector-icons";

export default function RecordScreen() {
  const cameraRef = useRef<CameraView>(null);

  const [permission, requestPermission] = useCameraPermissions();

  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    requestPermission();
  }, []);

  if (!permission?.granted) {
    return (
      <View className="flex-1 items-center justify-center bg-white">
        <Text className="mb-3 text-gray-700">
          Camera access is required
        </Text>

        <TouchableOpacity
          onPress={requestPermission}
          className="bg-[#1A3263] px-5 py-3 rounded-xl"
        >
          <Text className="text-white font-outfit-bold">
            Allow Camera
          </Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Pick video from gallery
  async function pickVideo() {
    setLoading(true);

    const result =
      await ImagePicker.launchImageLibraryAsync({
        mediaTypes:
          ImagePicker.MediaTypeOptions.Videos,
      });

    setLoading(false);

    if (!result.canceled) {
      console.log("Selected:", result.assets[0].uri);

      // TODO: Upload here
    }
  }

  // Record video
  async function startRecording() {
    if (!cameraRef.current) return;

    setRecording(true);

    const video =
      await cameraRef.current.recordAsync();

    setRecording(false);

    console.log("Recorded:", video.uri);

    // TODO: Upload here
  }

  function stopRecording() {
    cameraRef.current?.stopRecording();
  }

  return (
    <View className="flex-1 bg-black">
      {/* Camera */}
      <CameraView
        ref={cameraRef}
        className="flex-1"
        facing="back"
        mode="video"
      />

      {/* Controls */}
      <View className="absolute bottom-0 w-full px-6 pb-10 pt-6 bg-black/40">
        <View className="flex-row items-center justify-between">
          {/* Upload */}
          <TouchableOpacity
            onPress={pickVideo}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="white" />
            ) : (
              <Ionicons
                name="cloud-upload-outline"
                size={30}
                color="white"
              />
            )}
          </TouchableOpacity>

          {/* Record Button */}
          <TouchableOpacity
            onPress={
              recording
                ? stopRecording
                : startRecording
            }
            className={`w-20 h-20 rounded-full items-center justify-center ${
              recording
                ? "bg-red-600"
                : "bg-white"
            }`}
          >
            <View
              className={`w-14 h-14 rounded-full ${
                recording
                  ? "bg-red-800"
                  : "bg-red-500"
              }`}
            />
          </TouchableOpacity>

          {/* Flip */}
          <TouchableOpacity>
            <Ionicons
              name="camera-reverse"
              size={30}
              color="white"
            />
          </TouchableOpacity>
        </View>

        <Text className="text-center text-white text-xs mt-3 opacity-70">
          Hold to record • Upload from gallery
        </Text>
      </View>
    </View>
  );
}
