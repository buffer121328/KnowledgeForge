import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { ConfigProvider, App as AntdApp } from 'antd'
import antdZhCN from 'antd/locale/zh_CN'
import dayjs from 'dayjs'
import 'dayjs/locale/zh-cn'
import App from './App'
import './index.css'

dayjs.locale('zh-cn')

// Vite/ESM 下 antd locale 可能多包一层 default，未解包时表格空状态仍显示英文 No data
const zhCN = (antdZhCN as { default?: typeof antdZhCN }).default ?? antdZhCN

const theme = {
  token: {
    colorPrimary: '#155eef',
    colorSuccess: '#17b26a',
    colorWarning: '#f79009',
    colorError: '#f04438',
    colorBgLayout: '#f5f7fa',
    colorBorderSecondary: '#eaecf0',
    borderRadius: 8,
    borderRadiusLG: 12,
    fontSize: 14,
  },
  components: {
    Layout: {
      headerBg: '#fff',
      headerHeight: 72,
      headerPadding: '0 28px',
      bodyBg: '#f5f7fa',
      siderBg: '#101828',
    },
    Menu: {
      itemHeight: 44,
      itemBorderRadius: 8,
      subMenuItemBg: 'transparent',
    },
  },
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ConfigProvider locale={zhCN} theme={theme}>
      <AntdApp>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </AntdApp>
    </ConfigProvider>
  </StrictMode>,
)
