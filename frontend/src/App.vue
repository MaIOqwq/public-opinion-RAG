<template><div class="app-container"><div class="main-content"><ChatComponent /></div></div></template><script>import ChatComponent from './components/ChatComponent.vue'

export default {
  name: 'App',
  components: {
    ChatComponent
  }
}</script><style>.app-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: #f5f7fa;
}

.main-content {
  flex: 1;
  overflow: hidden;
}</style>