<template>
  <div class="app-wrapper">
    <div class="chat-container">
      <div class="main-content">
        <div class="header-bar">
          <div class="header-left">
            <div class="button-group">
              <el-tooltip content="新对话" placement="bottom">
                <el-button size="small" circle @click="newChat" class="new-chat-btn">
                  <el-icon><Plus /></el-icon>
                </el-button>
              </el-tooltip>
            </div>
          </div>
        </div>
        <div class="chat-container-wrapper" :class="{ expanded: hasSentMessage }">
          <div v-if="!hasSentMessage" class="chat-title">
            <img src="/logo2.png" alt="Logo" class="title-logo" />
            游戏舆情智能问答系统
          </div>
          <div class="chat-panel" :class="{ 'full-screen': hasSentMessage }">
            <div class="chat-history" ref="chatHistory">
              <div v-for="(msg, index) in currentChat.messages" :key="index"
                   :class="['message-item', msg.type]">
                <div class="message-avatar">{{ msg.type === 'user' ? '👤' : '🤖' }}</div>
                <div class="message-content">
                  <div class="message-text" v-html="formatContent(msg.content)"></div>
                  <div v-if="msg.sources && msg.sources.length" class="message-sources">
                    <el-collapse>
                      <el-collapse-item title="参考来源" name="sources">
                        <div class="source-item" v-for="(s, i) in msg.sources" :key="i">{{ s }}</div>
                      </el-collapse-item>
                    </el-collapse>
                  </div>
                  <div class="message-time">{{ msg.time }}</div>
                </div>
              </div>
              <div v-if="loading" class="loading-indicator">
                <div class="typing-indicator"><span></span><span></span><span></span></div>
              </div>
            </div>
            <div class="chat-input-area">
              <el-input
                v-model="inputMessage"
                type="textarea"
                :rows="2"
                placeholder="请输入您的问题..."
                :disabled="loading"
                @keyup.enter.exact="sendMessage"
              />
              <div class="input-actions">
                <el-button
                  type="primary"
                  @click="sendMessage"
                  :loading="loading"
                  :disabled="!inputMessage.trim() || loading"
                >发送</el-button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { Plus } from '@element-plus/icons-vue'

export default {
  name: 'ChatComponent',
  components: { Plus },
  data() {
    return {
      chatHistory: [{ title: '', messages: [], time: new Date().toLocaleString() }],
      currentChatIndex: 0,
      inputMessage: '',
      loading: false,
      hasSentMessage: false,
    }
  },
  computed: {
    currentChat() {
      return this.chatHistory[this.currentChatIndex]
    },
  },
  methods: {
    async sendMessage() {
      const message = this.inputMessage.trim()
      if (!message || this.loading) return

      this.hasSentMessage = true

      this.currentChat.messages.push({
        type: 'user',
        content: message,
        time: new Date().toLocaleTimeString(),
      })

      if (!this.currentChat.title) {
        this.currentChat.title = message.length > 30 ? message.substring(0, 30) + '...' : message
        this.currentChat.time = new Date().toLocaleString()
      }

      this.inputMessage = ''
      this.loading = true

      const sysMsg = { type: 'system', content: '', sources: [], time: new Date().toLocaleTimeString() }
      this.currentChat.messages.push(sysMsg)
      this.scrollToBottom()

      try {
        const response = await fetch('/api/rag/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: message }),
        })

        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })

          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = JSON.parse(line.slice(6))
              if (data.type === 'chunk') {
                sysMsg.content += data.content
              } else if (data.type === 'done') {
                sysMsg.sources = data.sources || []
                if (!sysMsg.content) sysMsg.content = data.answer || ''
              } else if (data.type === 'error') {
                sysMsg.content = data.content || '查询失败'
              }
              this.scrollToBottom()
            }
          }
        }
      } catch (error) {
        console.error('请求失败:', error)
        sysMsg.content = sysMsg.content || '请求失败，请稍后重试'
      } finally {
        this.loading = false
        this.scrollToBottom()
      }
    },

    formatContent(text) {
      if (!text) return ''
      return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>')
    },

    newChat() {
      this.chatHistory.push({ title: '', messages: [], time: new Date().toLocaleString() })
      this.currentChatIndex = this.chatHistory.length - 1
      this.hasSentMessage = false
    },

    scrollToBottom() {
      this.$nextTick(() => {
        const el = this.$refs.chatHistory
        if (el) el.scrollTop = el.scrollHeight
      })
    },
  },
  mounted() {
    this.scrollToBottom()
  },
}
</script>

<style scoped>
.app-wrapper {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: #ffffff;
}

.chat-container {
  flex: 1;
  display: flex;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.header-bar {
  padding: 12px 24px;
  display: flex;
  background-color: #ffffff;
  border-bottom: 1px solid #f0f0f0;
}

.button-group {
  display: flex;
  background-color: #f3f4f6;
  border-radius: 20px;
  padding: 2px;
}

.new-chat-btn {
  background-color: transparent;
  border: none;
  width: 36px;
  height: 36px;
  border-radius: 18px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.new-chat-btn:hover {
  background-color: rgba(0, 0, 0, 0.05);
}

.chat-container-wrapper {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px;
  transition: all 0.3s ease;
}

.chat-container-wrapper.expanded {
  padding: 0;
  justify-content: flex-start;
}

.chat-title {
  font-size: 24px;
  font-weight: 600;
  color: #374151;
  margin-bottom: 30px;
  display: flex;
  align-items: center;
  gap: 12px;
}

.title-logo {
  width: 60px;
  height: 60px;
  object-fit: contain;
}

.chat-panel {
  width: 100%;
  max-width: 800px;
  display: flex;
  flex-direction: column;
  background-color: #ffffff;
  overflow: hidden;
  min-height: 400px;
  transition: all 0.3s ease;
}

.chat-panel.full-screen {
  max-width: 100%;
  min-height: calc(100vh - 80px);
}

.chat-history {
  flex: 1;
  padding: 24px;
  overflow-y: auto;
  background-color: #ffffff;
}

.message-item {
  display: flex;
  margin-bottom: 24px;
  animation: fadeIn 0.3s ease;
}

.message-item.user { justify-content: flex-end; }
.message-item.system { justify-content: flex-start; }

.message-avatar {
  font-size: 28px;
  margin: 0 12px;
}

.message-content {
  max-width: 80%;
  padding: 16px 20px;
  border-radius: 12px;
  background-color: white;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.message-item.user .message-content {
  background-color: #3b82f6;
  color: white;
}

.message-item.system .message-content {
  background-color: white;
  color: #374151;
}

.message-text {
  line-height: 1.6;
  word-break: break-word;
  font-size: 15px;
}

.message-text code {
  background-color: #f3f4f6;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 13px;
}

.message-sources {
  margin-top: 12px;
}

.source-item {
  padding: 8px 12px;
  background-color: #f3f4f6;
  border-radius: 8px;
  margin-bottom: 6px;
  font-size: 12px;
  line-height: 1.5;
  color: #6b7280;
}

.message-time {
  font-size: 12px;
  color: #9ca3af;
  margin-top: 8px;
  text-align: right;
}

.loading-indicator {
  padding: 24px;
  display: flex;
  justify-content: center;
}

.typing-indicator {
  display: flex;
  gap: 8px;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #9ca3af;
  animation: typing 1.4s infinite ease-in-out both;
}

.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }

.chat-input-area {
  padding: 24px;
  background-color: white;
  border-top: 1px solid #e5e7eb;
}

.input-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 12px;
}

:deep(.el-textarea__inner) {
  resize: none;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes typing {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}

@media (max-width: 768px) {
  .header-bar { padding: 8px 16px; }
  .chat-container-wrapper { padding: 20px; }
  .chat-container-wrapper.expanded { padding: 0; }
  .chat-panel { margin: 0; }
  .chat-panel.full-screen { min-height: calc(100vh - 64px); }
  .message-content { max-width: 90%; }
  .title-logo { width: 48px; height: 48px; }
}
</style>
