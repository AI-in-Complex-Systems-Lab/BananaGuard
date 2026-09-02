import { useRef, useState } from 'react';
import { mediaUrl } from './api';
import { useAuth } from './AuthContext';


const MAX_DISPLAY_WIDTH = 720;


function BoxCorrectionModal({
  jobId,
  detection,
  onCancel,
  onSave,
}) {
  const { token } = useAuth();
  const overlayRef = useRef(null);

  const [label, setLabel] = useState(detection.label);
  const [box, setBox] = useState(detection.box);
  const [imageSize, setImageSize] = useState(null);
  const [imageError, setImageError] = useState(false);
  const [dragStart, setDragStart] = useState(null);
  const [dragCurrent, setDragCurrent] = useState(null);

  const frameUrl = mediaUrl(
    `/api/jobs/${jobId}/frames/${detection.frame}`,
    token
  );

  function handleImageLoad(event) {
    const naturalWidth = event.target.naturalWidth;
    const naturalHeight = event.target.naturalHeight;

    const displayWidth = Math.min(
      naturalWidth,
      MAX_DISPLAY_WIDTH
    );

    const displayHeight =
      (naturalHeight / naturalWidth) * displayWidth;

    setImageSize({
      naturalWidth,
      naturalHeight,
      displayWidth,
      displayHeight,
    });
  }

  function pointerPosition(event) {
    const rect = overlayRef.current.getBoundingClientRect();

    return {
      x: Math.min(
        Math.max(event.clientX - rect.left, 0),
        rect.width
      ),
      y: Math.min(
        Math.max(event.clientY - rect.top, 0),
        rect.height
      ),
    };
  }

  function handlePointerDown(event) {
    const position = pointerPosition(event);
    setDragStart(position);
    setDragCurrent(position);
  }

  function handlePointerMove(event) {
    if (!dragStart) return;
    setDragCurrent(pointerPosition(event));
  }

  function handlePointerUp() {
    if (dragStart && dragCurrent && imageSize) {
      const scale =
        imageSize.displayWidth / imageSize.naturalWidth;

      const x1 =
        Math.min(dragStart.x, dragCurrent.x) / scale;
      const y1 =
        Math.min(dragStart.y, dragCurrent.y) / scale;
      const x2 =
        Math.max(dragStart.x, dragCurrent.x) / scale;
      const y2 =
        Math.max(dragStart.y, dragCurrent.y) / scale;

      if (x2 - x1 >= 2 && y2 - y1 >= 2) {
        setBox([
          Math.round(x1 * 100) / 100,
          Math.round(y1 * 100) / 100,
          Math.round((x2 - x1) * 100) / 100,
          Math.round((y2 - y1) * 100) / 100,
        ]);
      }
    }

    setDragStart(null);
    setDragCurrent(null);
  }

  function resetBox() {
    setBox(detection.box);
  }

  function handleSave() {
    if (!label.trim()) return;
    onSave(label.trim(), box);
  }

  const scale = imageSize
    ? imageSize.displayWidth / imageSize.naturalWidth
    : 1;

  const boxStyle = {
    left: box[0] * scale,
    top: box[1] * scale,
    width: box[2] * scale,
    height: box[3] * scale,
  };

  const dragBoxStyle =
    dragStart && dragCurrent
      ? {
          left: Math.min(dragStart.x, dragCurrent.x),
          top: Math.min(dragStart.y, dragCurrent.y),
          width: Math.abs(dragCurrent.x - dragStart.x),
          height: Math.abs(dragCurrent.y - dragStart.y),
        }
      : null;

  return (
    <div style={styles.backdrop} onClick={onCancel}>
      <div
        style={styles.modal}
        onClick={(event) => event.stopPropagation()}
      >
        <h3 style={styles.title}>
          Correct Detection — frame {detection.frame}
        </h3>

        <p style={styles.hint}>
          Drag on the frame to draw a new box, or leave
          it as-is to only change the label.
        </p>

        <div
          ref={overlayRef}
          style={{
            ...styles.imageWrapper,
            width: imageSize?.displayWidth || '100%',
            height: imageSize?.displayHeight || 240,
          }}
          onMouseDown={
            imageSize ? handlePointerDown : undefined
          }
          onMouseMove={
            imageSize ? handlePointerMove : undefined
          }
          onMouseUp={
            imageSize ? handlePointerUp : undefined
          }
          onMouseLeave={
            imageSize ? handlePointerUp : undefined
          }
        >
          {imageError ? (
            <div style={styles.imageError}>
              Unable to load this frame. The source
              video may no longer be available.
            </div>
          ) : (
            <img
              src={frameUrl}
              alt={`Frame ${detection.frame}`}
              onLoad={handleImageLoad}
              onError={() => setImageError(true)}
              style={{
                width: imageSize?.displayWidth || '100%',
                height: 'auto',
                display: 'block',
                userSelect: 'none',
              }}
              draggable={false}
            />
          )}

          {imageSize && (
            <div
              style={{
                ...styles.detectionBox,
                ...boxStyle,
              }}
            />
          )}

          {dragBoxStyle && (
            <div
              style={{
                ...styles.dragBox,
                ...dragBoxStyle,
              }}
            />
          )}
        </div>

        <div style={styles.formRow}>
          <label style={styles.label}>
            Label
            <input
              type="text"
              value={label}
              onChange={(event) =>
                setLabel(event.target.value)
              }
              className="text-input"
            />
          </label>

          <button
            type="button"
            onClick={resetBox}
            className="btn"
          >
            Reset box
          </button>
        </div>

        <div style={styles.actions}>
          <button
            type="button"
            onClick={onCancel}
            className="btn btn-ghost"
          >
            Cancel
          </button>

          <button
            type="button"
            onClick={handleSave}
            disabled={!label.trim()}
            className="btn btn-primary"
          >
            Save correction
          </button>
        </div>
      </div>
    </div>
  );
}


const styles = {
  backdrop: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(3, 7, 18, 0.72)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
    padding: '20px',
  },

  modal: {
    background: 'var(--bg-panel)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)',
    padding: '24px',
    maxWidth: '800px',
    width: '100%',
    maxHeight: '90vh',
    overflowY: 'auto',
    color: 'var(--text-primary)',
  },

  title: {
    margin: 0,
  },

  hint: {
    color: 'var(--text-secondary)',
    marginTop: '6px',
  },

  imageWrapper: {
    position: 'relative',
    marginTop: '16px',
    background: '#000000',
    cursor: 'crosshair',
    maxWidth: '100%',
    borderRadius: 'var(--radius-md)',
    overflow: 'hidden',
  },

  imageError: {
    padding: '40px 20px',
    textAlign: 'center',
    color: '#f5f5f5',
  },

  detectionBox: {
    position: 'absolute',
    border: '2px solid #00ff66',
    pointerEvents: 'none',
  },

  dragBox: {
    position: 'absolute',
    border: '2px dashed var(--amber)',
    pointerEvents: 'none',
  },

  formRow: {
    display: 'flex',
    alignItems: 'flex-end',
    gap: '16px',
    marginTop: '18px',
  },

  label: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    fontWeight: 600,
    flex: 1,
  },

  actions: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '12px',
    marginTop: '22px',
  },
};


export default BoxCorrectionModal;
