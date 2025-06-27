import { createContext, useContext } from "react";

type FormContextType = {
    isLoading: boolean
}

const FormContext = createContext<FormContextType | null>(null)

export const useFormStatus = () => {
    const context = useContext(FormContext)
    if (!context) throw new Error('useFormStatus must be used inside formContainer')
    return context
}

export default FormContext