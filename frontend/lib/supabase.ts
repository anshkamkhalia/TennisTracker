import "react-native-url-polyfill/auto";

import { Platform } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";

import { createClient } from "@supabase/supabase-js";

const supabaseUrl =
  "https://vfagwujqxpxyrtxgquvr.supabase.co"

const supabaseKey =
  "sb_publishable_5X4jXwTbXCV2L7PXIF9unw_JiEOKhqJ";

// Pick correct storage
const storage =
  Platform.OS === "web"
    ? undefined // use default (localStorage)
    : AsyncStorage;

export const supabase = createClient(
  supabaseUrl,
  supabaseKey,
  {
    auth: {
      storage,
      autoRefreshToken: true,
      persistSession: true,
      detectSessionInUrl: false,
    },
  }
);


