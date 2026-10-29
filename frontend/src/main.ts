import { mount } from 'svelte'
import './app.css'
import App from './App.svelte'
import { unlockAudio } from './lib/audio'

// Browsers only allow audio once the page has had a user gesture
window.addEventListener('pointerdown', unlockAudio)
window.addEventListener('keydown', unlockAudio)

const app = mount(App, {
  target: document.getElementById('app')!,
})

export default app
