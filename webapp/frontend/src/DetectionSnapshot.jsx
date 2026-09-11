import { useState } from 'react';
import { mediaUrl } from './api';
import { useAuth } from './AuthContext';


const MAX_DISPLAY_WIDTH = 360;


/**
 * Read-only frame + box overlay, for side-by-side event inspection in
 * Model Comparison. Same image+overlay technique as
 * BoxCorrectionModal, without the drag-to-edit interaction — this is
 * for looking, not correcting.
 */
function DetectionSnapshot({ jobId, frame, box, label, score }) {
  const { token } = useAuth();

  const [imageSize, setImageSize] = useState(null);
  const [imageError, setImageError] = useState(false);

  const frameUrl = mediaUrl(
    `/api/jobs/${jobId}/frames/${frame}`,
    token
  );

  function handleImageLoad(event) {
    const naturalWidth = event.target.naturalWidth;
    const naturalHeight = event.target.naturalHeight;
    const displayWidth = Math.min(naturalWidth, MAX_DISPLAY_WIDTH);
    const displayHeight = (naturalHeight / naturalWidth) * displayWidth;

    setImageSize({ naturalWidth, displayWidth, displayHeight });
  }

  const scale = imageSize
    ? imageSize.displayWidth / imageSize.naturalWidth
    : 1;

  const boxStyle = box
    ? {
        left: box[0] * scale,
        top: box[1] * scale,
        width: box[2] * scale,
        height: box[3] * scale,
      }
    : null;

  return (
    <div className="detection-snapshot">
      <div
        className="detection-snapshot-image-wrapper"
        style={{
          width: imageSize?.displayWidth || '100%',
          height: imageSize?.displayHeight || 200,
        }}
      >
        {imageError ? (
          <div className="detection-snapshot-error">
            Unable to load this frame.
          </div>
        ) : (
          <img
            src={frameUrl}
            alt={`Frame ${frame}`}
            onLoad={handleImageLoad}
            onError={() => setImageError(true)}
            style={{
              width: imageSize?.displayWidth || '100%',
              height: 'auto',
              display: 'block',
            }}
            draggable={false}
          />
        )}

        {imageSize && boxStyle && (
          <div
            className="detection-snapshot-box"
            style={boxStyle}
          />
        )}
      </div>

      {label && (
        <div className="detection-snapshot-caption">
          {label}
          {typeof score === 'number' && ` (${score.toFixed(2)})`}
        </div>
      )}
    </div>
  );
}


export default DetectionSnapshot;
