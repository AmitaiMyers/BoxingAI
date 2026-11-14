// https://docs.expo.dev/guides/using-eslint/
// Same layering rules as the Self-Driving Racer project.
const { defineConfig } = require('eslint/config');
const expoConfig = require('eslint-config-expo/flat');

module.exports = defineConfig([
  expoConfig,
  {
    ignores: ['dist/*', '.expo/*', 'public/*'],
  },
  {
    // react-three-fiber's JSX elements (mesh, boxGeometry, ...) take three.js properties, not DOM ones.
    files: ['src/render/**/*.tsx'],
    rules: { 'react/no-unknown-property': 'off' },
  },
  {
    files: ['src/sim/**/*.ts'],
    ignores: ['src/sim/**/*.test.ts'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            { group: ['react', 'react-*', 'expo', 'expo-*', '@expo/*', '@react-three/*', 'three', 'zustand', 'zustand/*'], message: 'src/sim must stay pure TypeScript.' },
            { group: ['../engine/*', '../render/*', '../state/*', '../storage/*', '../ui/*'], message: 'src/sim must not depend on app layers.' },
          ],
        },
      ],
      // The simulation takes its randomness from a seeded generator and never reads a clock.
      'no-restricted-properties': [
        'error',
        { object: 'Math', property: 'random', message: 'Use a seeded generator (seededRandom in src/sim/bot.ts).' },
      ],
      'no-restricted-globals': [
        'error',
        { name: 'Date', message: 'The simulation never reads a clock.' },
        { name: 'performance', message: 'The simulation never reads a clock.' },
      ],
    },
  },
]);
