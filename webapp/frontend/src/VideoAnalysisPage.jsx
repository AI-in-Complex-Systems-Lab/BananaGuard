import { useState } from 'react';
import UploadPanel from './UploadPanel';
import JobHistoryPanel from './JobHistoryPanel';


const TABS = [
  { key: 'upload', label: 'New Upload' },
  { key: 'history', label: 'Past Videos' },
];

/**
 * Thin tab wrapper over the existing, unmodified UploadPanel and
 * JobHistoryPanel — "upload footage" and "look at past uploads" are
 * one task to a non-technical user ("analyze footage"), so they share
 * one nav destination instead of two.
 */
function VideoAnalysisPage({ initialTab, initialSelectedJobId }) {
  const [tab, setTab] = useState(initialTab || 'upload');

  return (
    <div>
      <div
        style={{
          display: 'flex',
          gap: 8,
          marginBottom: 20,
        }}
      >
        {TABS.map((option) => (
          <button
            key={option.key}
            type="button"
            className={`btn btn-sm ${
              tab === option.key ? 'btn-primary' : 'btn-ghost'
            }`}
            onClick={() => setTab(option.key)}
          >
            {option.label}
          </button>
        ))}
      </div>

      {tab === 'upload' && <UploadPanel />}

      {tab === 'history' && (
        <JobHistoryPanel
          initialSelectedJobId={initialSelectedJobId}
        />
      )}
    </div>
  );
}


export default VideoAnalysisPage;
