import { Text, TextProps } from "react-native";

export function AppText(props: TextProps) {
  return (
    <Text
      {...props}
      style={[
        { fontFamily: "Outfit_400Regular" },
        props.style,
      ]}
    />
  );
}
