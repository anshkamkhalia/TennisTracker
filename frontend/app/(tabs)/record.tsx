import { View, TouchableOpacity, ActivityIndicator } from "react-native";
import { useEffect, useRef, useState } from "react";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as ImagePicker from "expo-image-picker";
import { router } from "expo-router";

import { AppText as Text } from "@/components/AppText";
import { Ionicons } from "@expo/vector-icons";

import { processVideo } from "@/lib/api";

export default function RecordScreen() {
  const cameraRef = useRef<CameraView>(null);

  const [permission, requestPermission] =
    useCameraPermissions();

  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    requestPermission();
  }, []);

  /* ---------------- Permissions ---------------- */

  // Ask to access camera
  if (!permission?.granted) {
    return (
      <View className="flex-1 bg-white items-center justify-center px-6">
        <Text className="text-gray-700 mb-4 text-center">
          Camera access is required to record videos.
        </Text>

        <TouchableOpacity
          onPress={requestPermission}
          className="bg-[#1A3263] px-6 py-3 rounded-xl"
        >
          <Text className="text-white font-outfit-bold">
            Allow Camera
          </Text>
        </TouchableOpacity>
      </View>
    );
  }

  /* ---------------- Upload Helpers ---------------- */

  async function sendToServer(uri: string) {
    setLoading(true);

    try {
      const res = await processVideo(uri);

      if (!res?.url) {
        throw new Error("Invalid server response");
      }

      router.push({
        pathname: "./result",
        params: { url: res.url },
      });
    } catch (err: any) {
      alert(err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  /* ---------------- Pick From Gallery ---------------- */

  async function pickVideo() {
    const result =
      await ImagePicker.launchImageLibraryAsync({
        mediaTypes:
          ImagePicker.MediaTypeOptions.Videos,
      });

    if (!result.canceled) {
      await sendToServer(result.assets[0].uri);
    }
  }

  /* ---------------- Record Video ---------------- */

  async function startRecording() {
    if (!cameraRef.current || loading) return;

    setRecording(true);

    try {
      const video =
        await cameraRef.current.recordAsync();

      await sendToServer(video.uri);
    } catch (e) {
      console.log(e);
    } finally {
      setRecording(false);
    }
  }

  function stopRecording() {
    cameraRef.current?.stopRecording();
  }

  /* ---------------- UI ---------------- */

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
      <View className="absolute bottom-0 w-full px-6 pb-10 pt-6 bg-black/50">
        <View className="flex-row items-center justify-between">
          {/* Upload */}
          <TouchableOpacity
            onPress={pickVideo}
            disabled={loading}
          >
            <Ionicons
              name="cloud-upload-outline"
              size={30}
              color="white"
            />
          </TouchableOpacity>

          {/* Record Button */}
          <TouchableOpacity
            onPress={
              recording
                ? stopRecording
                : startRecording
            }
            disabled={loading}
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

          {/* Placeholder */}
          <View className="w-[30px]" />
        </View>

        <Text className="text-center text-white text-xs mt-3 opacity-70">
          Tap to record • Upload from gallery
        </Text>
      </View>

      {/* Loading Overlay */}
      {loading && (
        <View className="absolute inset-0 bg-black/70 items-center justify-center">
          <ActivityIndicator size="large" color="white" />

          <Text className="text-white mt-4">
            Processing video...
          </Text>

          <Text className="text-white/60 text-xs mt-1">
            This may take up to 1–2 minutes
          </Text>
        </View>
      )}
    </View>
  );
}
