import { useEffect, useMemo, useState } from 'react'
import './App.css'

const MAX_MESSAGES = 120
const MAX_ACTIVITIES = 8

function eventsUrl() {
  if (import.meta.env.VITE_EVENTS_URL) return import.meta.env.VITE_EVENTS_URL

  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const host = window.location.hostname || 'localhost'
  return `${protocol}://${host}:8765/events`
}

function formatTime(timestamp) {
  if (!timestamp) return ''
  return new Intl.DateTimeFormat([], { hour: 'numeric', minute: '2-digit' }).format(
    new Date(timestamp * 1000),
  )
}

function App() {
  const [connection, setConnection] = useState('connecting')
  const [messages, setMessages] = useState([])
  const [activities, setActivities] = useState([])

  const currentActivity = activities[0] || {
    label: connection === 'connected' ? 'Listening' : 'Waiting for backend',
    detail:
      connection === 'connected'
        ? 'Start speaking when the voice agent is ready.'
        : 'Run the Python agent with --dashboard, then refresh this page.',
  }

  useEffect(() => {
    let socket
    let reconnectTimer
    let disposed = false

    const connect = () => {
      setConnection('connecting')
      socket = new WebSocket(eventsUrl())

      socket.onopen = () => {
        if (!disposed) setConnection('connected')
      }

      socket.onmessage = (event) => {
        let payload
        try {
          payload = JSON.parse(event.data)
        } catch {
          return
        }

        if (payload.type === 'message') {
          setMessages((current) => [...current, payload].slice(-MAX_MESSAGES))
        }

        if (payload.type === 'activity') {
          setActivities((current) => [payload, ...current].slice(0, MAX_ACTIVITIES))
        }
      }

      socket.onerror = () => socket.close()
      socket.onclose = () => {
        if (disposed) return
        setConnection('disconnected')
        reconnectTimer = window.setTimeout(connect, 2500)
      }
    }

    connect()
    return () => {
      disposed = true
      window.clearTimeout(reconnectTimer)
      socket?.close()
    }
  }, [])

  const statusText = useMemo(() => {
    if (connection === 'connected') return 'Live'
    if (connection === 'connecting') return 'Connecting'
    return 'Reconnecting'
  }, [connection])

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">R</div>
          <div>
            <p className="eyebrow">VOICE ADVISOR</p>
            <h1>Riya</h1>
          </div>
        </div>
        <div className={`connection connection-${connection}`}>
          <span className="status-dot" aria-hidden="true" />
          {statusText}
        </div>
      </header>

      <section className="workspace" aria-label="Live voice conversation">
        <section className="conversation-panel">
          <div className="conversation-heading">
            <div>
              <p className="eyebrow">LIVE CONVERSATION</p>
              <h2>Insurance policy assistant</h2>
            </div>
            <span className="voice-note">Voice call</span>
          </div>

          <div className="messages" aria-live="polite">
            {messages.length === 0 ? (
              <div className="empty-state">
                <div className="wave" aria-hidden="true"><i /><i /><i /><i /><i /></div>
                <h3>Ready when you are</h3>
                <p>Your spoken words and Riya&apos;s replies will appear here during the call.</p>
              </div>
            ) : (
              messages.map((message, index) => (
                <article
                  className={`message message-${message.role}`}
                  key={`${message.timestamp}-${message.role}-${index}`}
                >
                  <p>{message.text}</p>
                  <time>{message.role === 'assistant' ? 'Riya' : 'You'} · {formatTime(message.timestamp)}</time>
                </article>
              ))
            )}
          </div>

          <div className="now-playing">
            <span className="pulse" aria-hidden="true" />
            <div>
              <strong>{currentActivity.label}</strong>
              <span>{currentActivity.detail}</span>
            </div>
          </div>
        </section>

        <aside className="activity-panel" aria-label="Call activity">
          <div className="activity-heading">
            <div>
              <p className="eyebrow">CALL ACTIVITY</p>
              <h2>Live status</h2>
            </div>
            <span className="activity-count">{activities.length}</span>
          </div>

          <ol className="activity-list">
            {activities.length === 0 ? (
              <li className="activity-empty">No activity yet. The panel will update without interrupting the call.</li>
            ) : (
              activities.map((activity, index) => (
                <li className="activity-item" key={`${activity.timestamp}-${index}`}>
                  <span className="activity-icon" aria-hidden="true">{index === 0 ? '●' : '○'}</span>
                  <div>
                    <strong>{activity.label}</strong>
                    <span>{activity.detail}</span>
                  </div>
                  <time>{formatTime(activity.timestamp)}</time>
                </li>
              ))
            )}
          </ol>

          <div className="privacy-note">
            <span aria-hidden="true">⌁</span>
            Live view only. Nothing is sent from this dashboard to the call.
          </div>
        </aside>
      </section>
    </main>
  )
}

export default App
