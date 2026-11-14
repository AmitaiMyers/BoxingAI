import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

import { COLORS } from '../src/render/palette';

export default function RootLayout() {
  return (
    <>
      <StatusBar style="light" hidden />
      <Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: COLORS.background } }} />
    </>
  );
}
