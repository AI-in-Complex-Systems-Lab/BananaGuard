import { useCallback, useEffect, useState } from 'react';

function listVideoInputs(deviceList) {
  return deviceList.filter((device) => device.kind === 'videoinput');
}

/**
 * Lists the browser-visible camera devices on this machine (built-in
 * webcam, any plugged-in USB cameras, etc.). Device labels and stable
 * ids are only available after the user has granted camera
 * permission at least once, so callers should prompt with
 * requestAccess() from a button click rather than on page load.
 */
function useCameraDevices() {
  const [devices, setDevices] = useState([]);
  const [permission, setPermission] = useState('unknown');

  useEffect(() => {
    let cancelled = false;

    async function loadDevices() {
      if (!navigator.mediaDevices?.enumerateDevices) return;

      const list = await navigator.mediaDevices.enumerateDevices();

      if (!cancelled) {
        setDevices(listVideoInputs(list));
      }
    }

    loadDevices();

    navigator.mediaDevices?.addEventListener?.(
      'devicechange',
      loadDevices
    );

    return () => {
      cancelled = true;

      navigator.mediaDevices?.removeEventListener?.(
        'devicechange',
        loadDevices
      );
    };
  }, []);

  const requestAccess = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
      });

      stream.getTracks().forEach((track) => track.stop());
      setPermission('granted');

      const list = await navigator.mediaDevices.enumerateDevices();
      setDevices(listVideoInputs(list));
    } catch (error) {
      console.error('Camera permission denied:', error);
      setPermission('denied');
    }
  }, []);

  const hasLabeledDevices = devices.some((device) => device.label);

  return {
    devices,
    permission,
    hasLabeledDevices,
    requestAccess,
  };
}


export default useCameraDevices;
