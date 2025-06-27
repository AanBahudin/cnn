import { configureStore } from "@reduxjs/toolkit";
import globalReducer from '@/cart/globalSlice'

export const store = configureStore({
    reducer: {
        globalState: globalReducer
    }
})