import {createSlice} from '@reduxjs/toolkit'

const defaultState = {
    selectedImg: '',
    uploadedImg: '',
    data: {type: 'tested'},
    processingData: {data: 'tested'}
}

const globalSlice = createSlice({
    name: 'global',
    initialState: defaultState,
    reducers: {
        setImage: (state, action) => {
            state.selectedImg = action.payload
        },
        removeImage: (state) => {
            state.selectedImg = ''
        },
        setData: (state, action) => {
            state.data = action.payload
        },
        setUploaded: (state, action) => {
            state.uploadedImg = action.payload
        },
        setProcessData: (state, action) => {
            state.processingData = action.payload
        }
    }
})

export default globalSlice.reducer
export const {setImage, removeImage, setData, setUploaded, setProcessData} = globalSlice.actions