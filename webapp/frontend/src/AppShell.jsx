import { useAuth } from './AuthContext';


const NAV_ITEMS = [
  { key: 'dashboard', icon: '▤', label: 'Dashboard' },
  { key: 'upload', icon: '↑', label: 'Process Video' },
  { key: 'webcam', icon: '●', label: 'Live Camera' },
  { key: 'history', icon: '≡', label: 'Job History' },
];

const ADMIN_NAV_ITEM = {
  key: 'admin',
  icon: '⚙',
  label: 'User Management',
};

const VIEW_TITLES = {
  dashboard: 'Dashboard',
  upload: 'Process a Video',
  webcam: 'Live Camera',
  history: 'Job History',
  admin: 'User Management',
};

function initials(name) {
  if (!name) return '?';

  return name
    .split(' ')
    .map((part) => part[0])
    .join('')
    .slice(0, 2)
    .toUpperCase();
}


function AppShell({ activeView, onNavigate, children }) {
  const { user, logout } = useAuth();

  const navItems =
    user?.role === 'admin'
      ? [...NAV_ITEMS, ADMIN_NAV_ITEM]
      : NAV_ITEMS;

  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div className="app-brand">
          <div className="app-brand-mark">BG</div>

          <div className="app-brand-text">
            <span className="app-brand-title">
              BananaGuard
            </span>

            <span className="app-brand-subtitle">
              Detection Platform
            </span>
          </div>
        </div>

        <div className="nav-group">
          <span className="nav-label">Operations</span>

          {navItems.map((item) => (
            <button
              key={item.key}
              type="button"
              className={`nav-item${
                activeView === item.key ? ' active' : ''
              }`}
              onClick={() => onNavigate(item.key)}
            >
              <span className="nav-icon">
                {item.icon}
              </span>

              {item.label}
            </button>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="user-chip">
            <div className="user-avatar">
              {initials(user?.display_name)}
            </div>

            <div className="user-meta">
              <span className="user-name">
                {user?.display_name}
              </span>

              <span className="user-role">
                {user?.role}
              </span>
            </div>
          </div>

          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={logout}
            style={{ width: '100%', marginTop: 8 }}
          >
            Sign Out
          </button>
        </div>
      </aside>

      <div className="app-main">
        <header className="app-topbar">
          <span className="app-topbar-title">
            {VIEW_TITLES[activeView] || ''}
          </span>

          <div className="app-topbar-user">
            <span className="badge badge-info">
              {user?.role}
            </span>
          </div>
        </header>

        <main className="app-content">{children}</main>
      </div>
    </div>
  );
}


export default AppShell;
