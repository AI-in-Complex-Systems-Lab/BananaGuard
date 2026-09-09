function StatusDot({ live }) {
  return (
    <span
      className={`camera-status-dot${live ? ' live' : ''}`}
      title={live ? 'Live' : 'Not connected'}
    />
  );
}

/**
 * One tile in the video-view grid. `camera.kind === 'browser'` is the
 * platform's one real, wired-up feed (this device's webcam, run
 * through live detection); every other kind is a placeholder for a
 * camera source that doesn't exist yet — real multi-camera ingestion
 * (RTSP/IP cameras) is future work, not built here.
 */
function CameraTile({
  camera,
  focused,
  onFocus,
  liveCamera,
}) {
  const isBrowserCamera = camera.kind === 'browser';
  const { refs, state, start, stop } = liveCamera;
  const { video: videoRef, canvas: canvasRef } = refs;
  const isLive = isBrowserCamera && state.isRunning;

  return (
    <div
      className={`camera-tile${focused ? ' focused' : ''}`}
      onClick={() => onFocus(camera.id)}
    >
      <div className="camera-tile-header">
        <span className="camera-tile-name">
          <StatusDot live={isLive} />
          {camera.name}
        </span>

        {isBrowserCamera && (
          <button
            type="button"
            className="btn btn-ghost btn-sm camera-tile-toggle"
            onClick={(event) => {
              event.stopPropagation();
              state.isRunning ? stop() : start();
            }}
          >
            {state.isRunning ? 'Stop' : 'Start'}
          </button>
        )}
      </div>

      <div className="camera-tile-body">
        {isBrowserCamera ? (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="camera-tile-video"
            />

            <canvas
              ref={canvasRef}
              width="640"
              height="480"
              className="camera-tile-canvas"
            />

            {!state.isRunning && (
              <div className="camera-tile-placeholder">
                <span>Camera stopped</span>
                <span className="camera-tile-placeholder-hint">
                  Click Start to begin live detection
                </span>
              </div>
            )}

            {state.isRunning && state.lastDetections.length > 0 && (
              <div className="camera-tile-alert">
                Alert: {state.lastDetections.length} detection
                {state.lastDetections.length === 1 ? '' : 's'}
              </div>
            )}
          </>
        ) : (
          <div className="camera-tile-placeholder">
            <span>No camera connected</span>
            <span className="camera-tile-placeholder-hint">
              Connect a camera system to enable this feed
            </span>
          </div>
        )}
      </div>
    </div>
  );
}


export default CameraTile;
