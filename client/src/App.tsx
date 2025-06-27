import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import KlasifikasiPage from './pages/KlasifikasiPage'

const App = () => {

  const router = createBrowserRouter([
    {
      path: '/',
      children: [
        {
          index: true,
          element: <KlasifikasiPage />
        }
      ]
    }
  ])

  return (
    <RouterProvider router={router} />
  )
}

export default App
