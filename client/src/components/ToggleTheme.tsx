import { Moon, Sun } from "lucide-react"
import { useTheme } from "@/provider/theme-provider"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"

const ToggleTheme = () => {

  const { theme, setTheme } = useTheme()

  const handleTheme = () => {
    let newTheme : 'light' | 'dark' = 'light'
    if (theme === 'light') newTheme = 'dark'
    setTheme(newTheme)
  }

  return (
    <div className="flex items-center space-x-2">
      <Switch id="airplane-mode" onCheckedChange={handleTheme} checked={theme === 'light'} />
      <Label htmlFor="airplane-mode">{theme === 'light' ? <Sun /> : <Moon />}</Label>
    </div>
  )
}

export default ToggleTheme