import {RouteObject} from 'react-router-dom'
import KlasifikasiPage from './pages/KlasifikasiPage'
import ProcessingData from './pages/ProcessingData'
 
const router : RouteObject[] = [
    {
        path: '/',
        children: [
            {
                index: true,
                element: <KlasifikasiPage />
            }
        ]
    }
]