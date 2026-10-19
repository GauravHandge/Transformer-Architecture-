import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

import path from 'path';

export default defineConfig({
	plugins: [sveltekit()],
	css: {
		preprocessorOptions: {
			scss: {
				includePaths: [path.resolve(__dirname, '.'), path.resolve(__dirname, 'src')],
				additionalData: `@use "${path.resolve(__dirname, 'src/styles/variables.scss').replace(/\\/g, '/')}" as *;\n`
			}
		}
	},
	server: {
		fs: {
			// Allow serving files from one level up to the project root
			allow: ['..']
		}
	}
});
