import { useState } from 'react';
import { useAuth } from './AuthContext';
import LoginPage from './LoginPage';
import AppShell from './AppShell';
import DashboardPage from './DashboardPage';
import UploadPanel from './UploadPanel';
import WebcamPanel from './WebcamPanel';
import JobHistoryPanel from './JobHistoryPanel';
import UsersAdminPage from './UsersAdminPage';
import SettingsPage from './SettingsPage';


function AuthenticatedApp({ user }) {
  const [activeView, setActiveView] =
    useState('dashboard');

  const [focusedJobId, setFocusedJobId] =
    useState(null);

  function handleNavigate(view) {
    setActiveView(view);

    if (view !== 'history') {
      setFocusedJobId(null);
    }
  }

  function handleOpenJob(jobId) {
    setFocusedJobId(jobId);
    setActiveView('history');
  }

  return (
    <AppShell
      activeView={activeView}
      onNavigate={handleNavigate}
    >
      {activeView === 'dashboard' && (
        <DashboardPage onOpenJob={handleOpenJob} />
      )}

      {activeView === 'upload' && <UploadPanel />}

      {activeView === 'webcam' && <WebcamPanel />}

      {activeView === 'history' && (
        <JobHistoryPanel
          initialSelectedJobId={focusedJobId}
        />
      )}

      {activeView === 'admin' &&
        user.role === 'admin' && <UsersAdminPage />}

      {activeView === 'settings' &&
        user.role === 'admin' && <SettingsPage />}
    </AppShell>
  );
}


function App() {
  const { isAuthenticated, initializing, user } =
    useAuth();

  if (initializing) {
    return (
      <div className="login-screen">
        <div className="card card-padded">
          Loading BananaGuard...
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginPage />;
  }

  return (
    <AuthenticatedApp key={user.username} user={user} />
  );
}


export default App;
