import { Component, type ReactNode } from 'react'

/** Shows a render error in the page instead of a blank white screen. Reset by changing `resetKey` (e.g. the route). */
export class ErrorBoundary extends Component<{ children: ReactNode; resetKey?: string }, { error: Error | null }> {
  state = { error: null as Error | null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidUpdate(prev: { resetKey?: string }) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null })
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="rounded-xl border border-rose-200 bg-rose-50 p-5 text-sm text-rose-800">
        <div className="mb-1 font-semibold">This page hit an error</div>
        <pre className="mb-3 whitespace-pre-wrap font-mono text-xs">{this.state.error.message}</pre>
        <button className="rounded-lg bg-rose-600 px-3 py-1.5 font-medium text-white hover:bg-rose-700"
          onClick={() => this.setState({ error: null })}>Try again</button>
      </div>
    )
  }
}
